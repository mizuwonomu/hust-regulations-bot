"""Đo tính đúng, thời gian và bộ nhớ của suy luận có và không có KV cache."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from llm_foundations.decoder_only_lm import DecoderOnlyLM
from llm_foundations.decoding import greedy_token
from llm_foundations.generation import generate
from llm_foundations.kv_cache.decoder_only_lm import CachedDecoderOnlyLM
from llm_foundations.kv_cache.generation import generation


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    max_seq_len: int = 256
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 2
    d_k: int = 16
    d_v: int = 24
    d_ff: int = 128
    eps: float = 1e-5

@dataclass(frozen=True)
class BenchmarkConfig:
    model: ModelConfig = ModelConfig()
    seed: int = 20261023
    dtype: str = "float32"
    device: str = "auto"
    warmup: int = 2
    repeats: int = 7
    prompt_lengths: tuple[int, ...] = (8, 32, 96)
    generated_tokens: int = 16
    generation_lengths: tuple[int, ...] = (1, 8, 32)
    generation_prompt_length: int = 32
    atol: float = 1e-5
    rtol: float = 1e-4

    def validate(self) -> None:
        m=self.model
        positive=(m.vocab_size,m.max_seq_len,m.d_model,m.num_layers,m.num_heads,m.d_k,m.d_v,m.d_ff)
        if any(x<1 for x in positive) or not math.isfinite(m.eps) or m.eps<=0: raise ValueError("model dimensions and eps must be positive")
        if self.device not in {"auto","cpu","cuda","cuda:0"}: raise ValueError("unsupported device")
        if self.dtype not in {"float32","float64"}: raise ValueError("dtype must be float32 or float64")
        if self.warmup<0 or self.repeats<1 or self.generated_tokens<0: raise ValueError("invalid warmup/repeats/token budget")
        if not self.prompt_lengths or any(p<1 or p>m.max_seq_len for p in self.prompt_lengths): raise ValueError("prompt lengths must fit model capacity")
        if not 1<=self.generation_prompt_length<=m.max_seq_len: raise ValueError("generation prompt must fit model capacity")
        if any(b<0 for b in self.generation_lengths): raise ValueError("generation budgets must be nonnegative")
        if not math.isfinite(self.atol) or not math.isfinite(self.rtol) or self.atol<0 or self.rtol<0: raise ValueError("tolerances must be finite and nonnegative")


def _mapping(ua: dict[str,torch.Tensor], ca: dict[str,torch.Tensor]) -> dict[str,str]:
    result={}
    for key in ua:
        target=key
        if key.startswith("blocks."):
            block,tail=key.split(".",2)[1:]
            pre=f"blocks.{block}."
            if tail.startswith("attention_residual.normalized_input."): target=pre+"normalize_input."+tail.rsplit(".",1)[-1]
            elif tail.startswith("attention_residual.multi_head_attention.w_o"): target=pre+"cached_mha.w_o"
            elif tail.startswith("attention_residual.multi_head_attention.heads."):
                part=tail.split("."); target=pre+f"cached_mha.cached_heads.{part[3]}.attention_head.{part[4]}"
            else: target=pre+tail
        if target not in ca: raise AssertionError(f"missing mapping {key} -> {target}")
        result[key]=target
    if len(result)!=len(ua) or set(result.values())!=set(ca): raise AssertionError("parameter mapping is not bijective")
    return result


def _norm_eps(m: nn.Module) -> list[float]:
    from llm_foundations.layer_norm import LayerNorm
    return [float(x.eps) for x in m.modules() if isinstance(x,LayerNorm)]

def build_matched_models(config: ModelConfig, *, device: torch.device|str) -> tuple[DecoderOnlyLM,CachedDecoderOnlyLM,dict[str,Any]]:
    torch.manual_seed(20261023)
    a=DecoderOnlyLM(**asdict(config)); b=CachedDecoderOnlyLM(**asdict(config))
    aa,bb=a.state_dict(),b.state_dict(); mapping=_mapping(aa,bb)
    with torch.no_grad():
        for src,dst in mapping.items():
            if aa[src].shape!=bb[dst].shape: raise AssertionError(f"shape mismatch {src} -> {dst}")
            bb[dst].copy_(aa[src])
    ap,bp=dict(a.named_parameters()),dict(b.named_parameters())
    if len(mapping)!=len(ap) or set(mapping)!=set(ap) or set(mapping.values())!=set(bp): raise AssertionError("named parameter coverage mismatch")
    for src,dst in mapping.items():
        if not torch.equal(ap[src],bp[dst]): raise AssertionError(f"parameter mismatch {src}")
    if _norm_eps(a)!=_norm_eps(b): raise AssertionError("LayerNorm eps mismatch")
    a.to(device); b.to(device)
    if next(a.parameters()).dtype!=next(b.parameters()).dtype or next(a.parameters()).device!=next(b.parameters()).device: raise AssertionError("dtype/device mismatch")
    return a,b,{"bijection":True,"count":len(mapping),"parameters":[{"source":x,"target":y} for x,y in mapping.items()],"layer_norm_eps":_norm_eps(a)}


def check_equivalence(uncached: DecoderOnlyLM,cached: CachedDecoderOnlyLM,prompt_ids: torch.Tensor,continuation_ids: torch.Tensor,*,atol:float,rtol:float)->dict[str,Any]:
    modes=[(x,x.training) for x in uncached.modules()]+[(x,x.training) for x in cached.modules()]; checks=[]
    try:
        uncached.eval();cached.eval()
        with torch.no_grad():
            full=uncached(prompt_ids); got,cache=cached.prefill(prompt_ids)
            checks.append(("prefill",full,got))
            for j,tok in enumerate(continuation_ids):
                full=uncached(torch.cat((prompt_ids,continuation_ids[:j+1])))[-1:]
                got,cache=cached.decode(tok.reshape(1),cache);checks.append((f"decode_{j}",full,got))
        records=[]
        for name,x,y in checks:
            err=float((x-y).abs().max().item());records.append({"name":name,"max_abs_logit_error":err,"finite":bool(torch.isfinite(x).all() and torch.isfinite(y).all()),"allclose":bool(torch.allclose(x,y,atol=atol,rtol=rtol))})
        return {"checks":records,"max_abs_logit_error":max((x["max_abs_logit_error"] for x in records),default=0.0),"finite":all(x["finite"] for x in records),"allclose":all(x["finite"] and x["allclose"] for x in records),"atol":atol,"rtol":rtol}
    finally:
        for mod,training in modes:mod.train(training)


def cache_nbytes(caches:list[list[tuple[torch.Tensor,torch.Tensor]]])->int:
    return sum(t.numel()*t.element_size() for layer in caches for pair in layer for t in pair)

def _sync(device:torch.device)->None:
    if device.type=="cuda":torch.cuda.synchronize(device)
def _timed(fn:Callable[[],Any],device:torch.device)->float:
    _sync(device);start=time.perf_counter();fn();_sync(device);return (time.perf_counter()-start)*1000
def _quant(values:list[float])->dict[str,float]|None:
    if not values:return None
    s=sorted(values)
    def q(p:float)->float:
        pos=(len(s)-1)*p;lo=int(pos);hi=min(lo+1,len(s)-1);return s[lo]+(s[hi]-s[lo])*(pos-lo)
    return {"median":statistics.median(s),"p25":q(.25),"p75":q(.75)}
def _hash(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _rand_ids(seed:int,n:int,cfg:ModelConfig,device:torch.device)->torch.Tensor:
    return torch.randint(cfg.vocab_size,(n,),generator=torch.Generator().manual_seed(seed),device="cpu").to(device)

def _call_arm(model:nn.Module,ids:torch.Tensor,budget:int,cached:bool)->torch.Tensor:
    return generation(model,ids,max_new_tokens=budget,selector=greedy_token) if cached else generate(model,ids,max_new_tokens=budget,selector=greedy_token)

def _memory_arm(cfg:ModelConfig,dtype:torch.dtype,device:torch.device,ids_cpu:torch.Tensor,budget:int,cached:bool,state:dict[str,torch.Tensor])->dict[str,Any]:
    if device.type!="cuda":return {"available":False,"reason":"CUDA allocator metrics unavailable on CPU","pre_model_allocated_mib":None,"baseline_allocated_mib":None,"peak_allocated_mib":None,"peak_delta_mib":None,"ending_allocated_mib":None}
    torch.cuda.synchronize(device); pre=torch.cuda.memory_allocated(device)
    model=(CachedDecoderOnlyLM if cached else DecoderOnlyLM)(**asdict(cfg))
    model.load_state_dict(state)
    model.to(device=device,dtype=dtype).eval()
    ids=ids_cpu.to(device)
    baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
    out=_call_arm(model,ids,budget,cached);torch.cuda.synchronize(device)
    peak=torch.cuda.max_memory_allocated(device);ending=torch.cuda.memory_allocated(device)
    ans={"available":True,"pre_model_allocated_mib":pre/2**20,"baseline_allocated_mib":baseline/2**20,"peak_allocated_mib":peak/2**20,"peak_delta_mib":(peak-baseline)/2**20,"ending_allocated_mib":ending/2**20,"boundary":"one active model + prompt at baseline; baseline includes the separately reported pre-model allocated footprint; wrapper output live at end; max_memory_allocated over actual public generation; reserved memory excluded"}
    del out,ids,model;gc.collect();torch.cuda.synchronize(device)
    return ans

def _source_metadata()->dict[str,Any]:
    root=Path(__file__).resolve().parents[2]
    relatives=[Path("llm_foundations/experiments/kv_cache_benchmark.py"),Path("llm_foundations/experiments/test_kv_cache_benchmark.py"),Path("llm_foundations/experiments/kv_cache_benchmark.ipynb"),Path("llm_foundations/decoder_only_lm.py"),Path("llm_foundations/kv_cache/decoder_only_lm.py"),Path("llm_foundations/generation.py"),Path("llm_foundations/kv_cache/generation.py"),Path("llm_foundations/decoding.py"),Path("llm_foundations/decoder_block.py"),Path("llm_foundations/kv_cache/decoder_block.py"),Path("llm_foundations/multi_head_attention.py"),Path("llm_foundations/kv_cache/multi_head_attention.py"),Path("llm_foundations/embeddings/input_embedding.py"),Path("llm_foundations/layer_norm.py"),Path("llm_foundations/lm_head.py"),Path("llm_foundations/ffn.py")]
    try:rev=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:rev=None
    return {"revision":rev,"sha256":{p.as_posix():_hash(root/p) for p in relatives if (root/p).exists()}}

def _phase_samples(a:DecoderOnlyLM,b:CachedDecoderOnlyLM,ids:torch.Tensor,continuation:torch.Tensor,n:int,repeats:int,warmup:int,device:torch.device)->dict[str,list[float]|None]:
    modes=[(m,m.training) for m in a.modules()]+[(m,m.training) for m in b.modules()]
    d=max(n-1,0);out={"uncached_prefill_ms":[],"cached_prefill_ms":[],"uncached_decode_total_ms":[],"cached_decode_total_ms":[]}
    if n==0:return {k:None for k in out}
    if d==0:out["uncached_decode_total_ms"]=None;out["cached_decode_total_ms"]=None
    def pref_u():a(ids)
    def pref_c():b.prefill(ids)
    def dec_u():
        for j in range(d):a(torch.cat((ids,continuation[:j+1])))
    def dec_c(cache):
        for j in range(d):cache=b.decode(continuation[j:j+1],cache)[1]
    try:
        a.eval();b.eval()
        with torch.no_grad():
            for _ in range(warmup):
                pref_u();pref_c()
                if d:
                    dec_u();warm_logits,c=b.prefill(ids);del warm_logits;dec_c(c)
            for r in range(repeats):
                arms=("u","c") if r%2==0 else ("c","u")
                for arm in arms:
                    if arm=="u":out["uncached_prefill_ms"].append(_timed(pref_u,device))
                    else:out["cached_prefill_ms"].append(_timed(pref_c,device))
                    if d:
                        if arm=="u":out["uncached_decode_total_ms"].append(_timed(dec_u,device))
                        else:
                            # Tạo cache mới và prefill nằm ngoài khoảng đo decode
                            prefill_logits,initial=b.prefill(ids);del prefill_logits;_sync(device);start=time.perf_counter();dec_c(initial);_sync(device);out["cached_decode_total_ms"].append((time.perf_counter()-start)*1000)
    finally:
        for mod,training in modes:mod.train(training)
    return out


def _observe(cases:list[dict[str,Any]])->str:
    usable=[x for x in cases if x["sweep"]=="prompt" and x["actual_tokens"]>0]
    if not usable:return "Không có token mới nào được sinh; không thể so sánh tốc độ generation."
    ratios=[x["speedups"].get("generation_ms") for x in usable if x["speedups"].get("generation_ms") is not None]
    if not ratios:return "Không có số đo generation hợp lệ để kết luận xu hướng."
    med=statistics.median(ratios)
    direction="nhanh hơn" if med>1 else "chậm hơn" if med<1 else "ngang nhau"
    return f"Trong prompt sweep đã đo, median speedup ratio-of-medians của wrapper cache là {med:.3f}x (cache {direction}); đây là mô hình chưa huấn luyện, chỉ mô tả workload nhỏ này và không đại diện chất lượng hay production."


def run_benchmark(config:BenchmarkConfig)->dict[str,Any]:
    config.validate()
    device=torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if config.device=="auto" else config.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    dtype=getattr(torch,config.dtype)
    a,b,mapping=build_matched_models(config.model,device=device);a.to(dtype);b.to(dtype)
    # Bản sao CPU giữ trọng số đã khớp mà không giữ arm còn lại trên CUDA
    state_a={k:v.detach().cpu().clone() for k,v in a.state_dict().items()}
    keymap={entry["source"]:entry["target"] for entry in mapping["parameters"]}
    state_b={dst:b.state_dict()[dst].detach().cpu().clone() for dst in keymap.values()}
    # Workload độc lập: sweep mặc định chỉ dùng chung chiều được giữ cố định
    workloads=[]
    for p in config.prompt_lengths:workloads.append(("prompt",p,config.generated_tokens))
    for budget in config.generation_lengths:workloads.append(("budget",config.generation_prompt_length,budget))
    unique=[];seen=set()
    for sweep,p,budget in workloads:
        key=(sweep,p,budget)
        if key not in seen:seen.add(key);unique.append(key)
    cases=[];correctness=[];memory=[]
    for ix,(sweep,p,budget) in enumerate(unique):
        n=min(budget,config.model.max_seq_len-p);d=max(n-1,0)
        prompt_seed=config.seed+(3101 if sweep=="prompt" else 4701)
        continuation_seed=config.seed+10291
        prompt=_rand_ids(prompt_seed,p,config.model,device);cont=_rand_ids(continuation_seed,d,config.model,device)
        if n:
            eq=check_equivalence(a,b,prompt,cont,atol=config.atol,rtol=config.rtol)
            if not eq["allclose"]:raise AssertionError("correctness gate failed; refusing interpreted benchmark")
        else:eq={"checks":[],"max_abs_logit_error":None,"finite":None,"allclose":None,"atol":config.atol,"rtol":config.rtol,"skipped":"N=0 performs no model calls"}
        correctness.append({"sweep":sweep,"prompt_length":p,"requested_tokens":budget,"actual_tokens":n,**eq})
        phase=_phase_samples(a,b,prompt,cont,n,config.repeats,config.warmup,device)
        metrics={**phase,"uncached_generation_ms":[],"cached_generation_ms":[],"uncached_tokens_per_second":[],"cached_tokens_per_second":[]}
        if n==0:
            # Wrapper trả nguyên IDs mà không gọi model; phép đo no-op được gắn nhãn riêng
            for _ in range(config.repeats):
                iu=prompt.clone();metrics["uncached_generation_ms"].append(_timed(lambda iu=iu:_call_arm(a,iu,budget,False),device))
                ic=prompt.clone();metrics["cached_generation_ms"].append(_timed(lambda ic=ic:_call_arm(b,ic,budget,True),device))
            del iu,ic
            metrics["uncached_tokens_per_second"]=[None]*config.repeats;metrics["cached_tokens_per_second"]=[None]*config.repeats
        else:
            for _ in range(config.warmup):_call_arm(a,prompt.clone(),budget,False);_call_arm(b,prompt.clone(),budget,True)
            for r in range(config.repeats):
                arms=("u","c") if r%2==0 else ("c","u")
                for arm in arms:
                    model=a if arm=="u" else b;input_ids=prompt.clone()
                    call=lambda model=model,input_ids=input_ids,arm=arm:_call_arm(model,input_ids,budget,arm=="c")
                    ms=_timed(call,device);name="uncached" if arm=="u" else "cached";metrics[name+"_generation_ms"].append(ms);metrics[name+"_tokens_per_second"].append(n/(ms/1000))
            del call,input_ids
        # Output wrapper và payload cache được kiểm tra riêng ngoài mẫu latency
        with torch.no_grad():
            out_a=_call_arm(a,prompt.clone(),budget,False);out_b=_call_arm(b,prompt.clone(),budget,True)
            agrees=bool(torch.equal(out_a,out_b));output_len=int(out_a.numel())
            output_ids_uncached=out_a.detach().cpu().tolist();output_ids_cached=out_b.detach().cpu().tolist()
            if n:
                prefill_logits,tmp=b.prefill(prompt);del prefill_logits
                for tok in cont[:d]:
                    decode_logits,tmp=b.decode(tok.reshape(1),tmp);del decode_logits
                if d:del tok
                T=p+d;actual_kv=cache_nbytes(tmp);del tmp
                theoretical=config.model.num_layers*config.model.num_heads*T*(config.model.d_k+config.model.d_v)*torch.empty((),dtype=dtype).element_size()
            else:actual_kv=None;theoretical=None
        prompt_ids_list=prompt.detach().cpu().tolist();continuation_ids_list=cont.detach().cpu().tolist()
        input_hash=hashlib.sha256(prompt.detach().cpu().numpy().tobytes()+cont.detach().cpu().numpy().tobytes()).hexdigest()
        prompt_cpu=prompt.detach().cpu().clone()
        del out_a,out_b
        if device.type=="cuda":
            a.to("cpu");b.to("cpu");del prompt,cont;gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache()
        mem_a=_memory_arm(config.model,dtype,device,prompt_cpu,budget,False,state_a)
        if device.type=="cuda":torch.cuda.empty_cache()
        mem_b=_memory_arm(config.model,dtype,device,prompt_cpu,budget,True,state_b)
        if device.type=="cuda":
            torch.cuda.empty_cache();a.to(device);b.to(device)
        del prompt_cpu;gc.collect()
        sample_summary={k:_quant([x for x in v if x is not None]) if v is not None else None for k,v in metrics.items()}
        speedups={}
        for metric in ("prefill_ms","decode_total_ms","generation_ms"):
            u=sample_summary.get("uncached_"+metric);c=sample_summary.get("cached_"+metric)
            speedups[metric]=(u["median"]/c["median"] if u and c and c["median"] else None)
        if sample_summary["uncached_decode_total_ms"] and d:
            for arm in ("uncached","cached"):
                q=sample_summary[arm+"_decode_total_ms"]
                sample_summary[arm+"_decode_ms_per_step"]={k:v/d for k,v in q.items()}
        cases.append({"case_id":f"{sweep}-p{p}-b{budget}","sweep":sweep,"held_fixed":{"prompt_sweep_budget":config.generated_tokens if sweep=="prompt" else None,"budget_sweep_prompt":config.generation_prompt_length if sweep=="budget" else None},"prompt_length":p,"requested_tokens":budget,"actual_tokens":n,"decode_steps":d,"context_lengths_uncached_decode":[p+j+1 for j in range(d)],"context_lengths_cached_incoming":[p+j for j in range(d)],"final_cache_length":p+d if n else None,"prompt_ids":prompt_ids_list,"continuation_ids":continuation_ids_list,"input_ids_sha256":input_hash,"output_ids_uncached":output_ids_uncached,"output_ids_cached":output_ids_cached,"output_length":output_len,"generation_ids_agree":agrees,"samples":metrics,"summaries":sample_summary,"speedups":speedups,"cache":{"actual_kv_bytes":actual_kv,"kv_mib":actual_kv/2**20 if actual_kv is not None else None,"theoretical_bytes":theoretical,"theoretical_kv_mib":theoretical/2**20 if theoretical is not None else None},"memory":{"uncached":mem_a,"cached":mem_b}})
        gc.collect()
    del a,b;gc.collect()
    return {"schema_version":2,"config":asdict(config),"environment":{"python":sys.version,"torch":torch.__version__,"cuda_version":torch.version.cuda,"device":str(device),"gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None,"threads":torch.get_num_threads(),"tf32_matmul":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn":torch.backends.cudnn.allow_tf32,"warmup":config.warmup,"repeats":config.repeats,"weight_seed":20261023,"input_seed":config.seed,"source":_source_metadata()},"mapping":mapping,"correctness":correctness,"cases":cases,"memory_scope":"separate pass per arm; competing models moved to CPU; one active model and prompt at baseline; pre-model allocated footprint is recorded and included in baseline; allocator allocated bytes, not reserved","metric_definitions":{"quantile":"linear interpolation at position (n-1)*p","speedup":"ratio of medians for the same case and metric; >1 favors cached","prefill_ms":"full prompt call, output and cache creation included, selector excluded","decode_total_ms":"D decode calls only; prompt computation/cache preparation outside interval; no per-token synchronization","decode_ms_per_step":"decode_total_ms / D; null for D=0","generation_ms":"actual public wrapper wall time, synchronized at interval boundaries; prompt.clone outside interval","tokens_per_second":"actual generated token count / wrapper seconds; null for N=0","kv_mib":"live K/V payload bytes / 2^20 at T=P+D; not allocator total"},"observations_vi":_observe(cases)}


def save_results(result:dict[str,Any],output_root:str|Path)->Path:
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True);run=root/(dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]);run.mkdir()
    (run/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,default=str)+"\n")
    definitions={"uncached_prefill_ms":"one full uncached prompt forward, wall milliseconds","cached_prefill_ms":"cached prefill including logits/cache return, wall milliseconds","uncached_decode_total_ms":"D full-prefix uncached forwards, decode-only wall milliseconds","cached_decode_total_ms":"D cached decode calls after external prefill, decode-only wall milliseconds","uncached_generation_ms":"public uncached wrapper wall milliseconds","cached_generation_ms":"public cached wrapper wall milliseconds","uncached_tokens_per_second":"actual generated tokens divided by uncached wrapper seconds","cached_tokens_per_second":"actual generated tokens divided by cached wrapper seconds"}
    raw=[];summ=[]
    for c in result["cases"]:
        common={"case_id":c["case_id"],"sweep":c["sweep"],"prompt_length":c["prompt_length"],"requested_tokens":c["requested_tokens"],"actual_tokens":c["actual_tokens"]}
        for metric,vals in c["samples"].items():
            unit="tokens/s" if "per_second" in metric else "ms"
            values=vals if vals is not None else [None]
            reason=None if vals is not None else ("N=0: phase not applicable" if c["actual_tokens"]==0 else "N=1: decode phase not applicable")
            for ix,val in enumerate(values):raw.append({**common,"metric":metric,"definition":definitions.get(metric,metric),"sample_index":ix if vals is not None else None,"value":val,"unit":unit,"unavailable_reason":reason if reason is not None else ("N=0: throughput undefined" if val is None and "per_second" in metric else None)})
        for metric,stats in c["summaries"].items():
            unit="tokens/s" if "per_second" in metric else "ms"
            if stats:summ.append({**common,"metric":metric,"definition":definitions.get(metric,metric),"unit":unit,**stats,"unavailable_reason":None})
            else:summ.append({**common,"metric":metric,"definition":definitions.get(metric,metric),"unit":unit,"median":None,"p25":None,"p75":None,"unavailable_reason":"N=0 or N=1 has no applicable decode measurement"})
        for metric,ratio in c["speedups"].items():summ.append({**common,"metric":"speedup_"+metric,"definition":"ratio of uncached median / cached median, same case and scope; no quartile ratio","unit":"ratio","median":ratio,"p25":None,"p75":None,"unavailable_reason":None if ratio is not None else "metric unavailable"})
        for arm in ("uncached","cached"):
            for metric in ("pre_model_allocated_mib","baseline_allocated_mib","peak_allocated_mib","peak_delta_mib","ending_allocated_mib"):
                val=c["memory"][arm].get(metric)
                summ.append({**common,"metric":arm+"_"+metric,"definition":"isolated-arm CUDA allocator measurement; allocated, not reserved","unit":"MiB","median":val,"p25":None,"p75":None,"unavailable_reason":c["memory"][arm].get("reason")})
        for metric in ("kv_mib","theoretical_kv_mib"):
            summ.append({**common,"metric":metric,"definition":"logical K/V payload; excludes allocator peak","unit":"MiB","median":c["cache"].get(metric),"p25":None,"p75":None,"unavailable_reason":"N=0 has no cache" if c["cache"].get(metric) is None else None})
    for name,rows in (("raw.csv",raw),("summary.csv",summ)):
        if rows:
            with (run/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
    return run


def plot_results(result:dict[str,Any],*,output_dir:str|Path|None=None)->dict[str,Any]:
    import matplotlib.pyplot as plt
    figs={};cases=result["cases"]
    def plot(key:str,title:str,ylabel:str,sweep:str,metric:str,ratio:bool=False,throughput:bool=False):
        fig,ax=plt.subplots()
        selected=[c for c in cases if c["sweep"]==sweep and (c["speedups"].get(metric) is not None if ratio else c["summaries"].get(metric))]
        selected.sort(key=lambda c:c["prompt_length"] if sweep=="prompt" else c["requested_tokens"])
        x=[c["prompt_length"] if sweep=="prompt" else c["requested_tokens"] for c in selected]
        if throughput:metrics=[c["summaries"].get(metric) for c in selected]
        elif ratio:metrics=[c["speedups"].get(metric) for c in selected]
        else:metrics=[c["summaries"].get(metric) for c in selected]
        good=[(xx,ss) for xx,ss in zip(x,metrics) if ss is not None]
        if good:
            xx=[p[0] for p in good];ss=[p[1] for p in good]
            if ratio: y=ss;lo=hi=None
            else:y=[z["median"] for z in ss];lo=[z["median"]-z["p25"] for z in ss];hi=[z["p75"]-z["median"] for z in ss]
            ax.plot(xx,y,marker="o",label=metric)
            if lo is not None:ax.fill_between(xx,[v-e for v,e in zip(y,lo)],[v+e for v,e in zip(y,hi)],alpha=.2,label="p25-p75")
        if ratio:ax.axhline(1,color="black",linestyle="--",label="speedup = 1")
        ax.set(title=title+"\n"+("B="+str(result["config"]["generated_tokens"]) if sweep=="prompt" else "P="+str(result["config"]["generation_prompt_length"])),xlabel="Prompt tokens" if sweep=="prompt" else "Requested new tokens",ylabel=ylabel);ax.legend();fig.tight_layout();figs[key]=fig
        if output_dir:fig.savefig(Path(output_dir)/(key+".svg"),format="svg")
    root=Path(output_dir) if output_dir else None
    if root:root.mkdir(parents=True,exist_ok=True)
    plot("prompt_prefill_uncached","Prompt sweep: uncached prefill","ms","prompt","uncached_prefill_ms")
    plot("prompt_prefill_cached","Prompt sweep: cached prefill","ms","prompt","cached_prefill_ms")
    plot("prompt_decode_uncached","Prompt sweep: uncached controlled decode average","ms / step","prompt","uncached_decode_ms_per_step")
    plot("prompt_decode_cached","Prompt sweep: cached controlled decode average","ms / step","prompt","cached_decode_ms_per_step")
    plot("prompt_generation_uncached","Prompt sweep: wrapper generation","ms","prompt","uncached_generation_ms")
    plot("prompt_generation_cached","Prompt sweep: wrapper generation","ms","prompt","cached_generation_ms")
    plot("prompt_speedup","Prompt sweep: wrapper speedup","ratio","prompt","generation_ms",ratio=True)
    plot("budget_generation_uncached","Budget sweep: wrapper generation","ms","budget","uncached_generation_ms")
    plot("budget_generation_cached","Budget sweep: wrapper generation","ms","budget","cached_generation_ms")
    plot("budget_speedup","Budget sweep: wrapper speedup","ratio","budget","generation_ms",ratio=True)
    # Throughput có trục riêng và N=0 được ghi N/A
    for arm in ("uncached","cached"):
        fig,ax=plt.subplots();sel=sorted([c for c in cases if c["sweep"]=="prompt"],key=lambda c:c["prompt_length"]);x=[];y=[]
        for c in sel:
            s=c["summaries"].get(arm+"_tokens_per_second")
            if s:x.append(c["prompt_length"]);y.append(s["median"])
        if x:
            stats=[c["summaries"][arm+"_tokens_per_second"] for c in sel if c["summaries"].get(arm+"_tokens_per_second")]
            ax.plot(x,y,marker="o",label=arm)
            ax.fill_between(x,[s["p25"] for s in stats],[s["p75"] for s in stats],alpha=.2,label="p25-p75")
        ax.set(title=f"Prompt sweep: wrapper throughput (B={result['config']['generated_tokens']})",xlabel="Prompt tokens",ylabel="Generated tokens/s");ax.legend();fig.tight_layout();key="prompt_throughput_"+arm;figs[key]=fig
        if root:fig.savefig(root/(key+".svg"),format="svg")
    # Biểu đồ bộ nhớ tách payload khỏi số đo allocator
    for name,ykey,ylabel in (("kv_payload","kv_mib","Logical K/V payload (MiB)"),("cuda_peak","peak_allocated_mib","CUDA peak allocated (MiB)"),("cuda_peak_delta","peak_delta_mib","CUDA peak delta over model+prompt baseline (MiB)")):
        fig,ax=plt.subplots();sel=sorted([c for c in cases if c["sweep"]=="prompt"],key=lambda c:c["prompt_length"])
        if name=="kv_payload":
            xy=[(c["prompt_length"],c["cache"][ykey]) for c in sel if c["cache"][ykey] is not None]
            if xy:ax.plot([x for x,_ in xy],[y for _,y in xy],marker="o",label="K/V payload")
        else:
            for arm in ("uncached","cached"):
                xy=[(c["prompt_length"],c["memory"][arm][ykey]) for c in sel if c["memory"][arm][ykey] is not None]
                if xy:ax.plot([x for x,_ in xy],[y for _,y in xy],marker="o",label=arm)
        if not ax.lines:ax.text(.5,.5,"N/A: CUDA unavailable or no generated cache",ha="center",va="center",transform=ax.transAxes)
        ax.set(title=name.replace("_"," ")+f" (prompt sweep B={result['config']['generated_tokens']})",xlabel="Prompt tokens",ylabel=ylabel);ax.legend();fig.tight_layout();figs[name]=fig
        if root:fig.savefig(root/(name+".svg"),format="svg")
    return figs


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--output-root",required=True);p.add_argument("--smoke",action="store_true");args=p.parse_args()
    cfg=BenchmarkConfig(warmup=1,repeats=2,prompt_lengths=(8,16),generated_tokens=4,generation_lengths=(1,4),generation_prompt_length=8) if args.smoke else BenchmarkConfig()
    result=run_benchmark(cfg);path=save_results(result,args.output_root);plot_results(result,output_dir=path/"graphs");print(json.dumps({"run":str(path),"device":result["environment"]["device"],"correct":all(x["allclose"] for x in result["correctness"]),"observations_vi":result["observations_vi"]},ensure_ascii=False))
if __name__=="__main__":main()
