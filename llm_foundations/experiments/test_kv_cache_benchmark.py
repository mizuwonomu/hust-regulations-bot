"""Kiểm thử các hợp đồng số học và đối chiếu benchmark KV cache."""
import torch
from llm_foundations.experiments.kv_cache_benchmark import (
    BenchmarkConfig, ModelConfig, _phase_samples, _quant, build_matched_models,
    cache_nbytes, check_equivalence, run_benchmark, save_results, plot_results,
)
from pathlib import Path


def test_mapping_and_real_logits():
    cfg=ModelConfig(vocab_size=17,max_seq_len=20,d_model=12,num_layers=2,num_heads=2,d_k=3,d_v=5,d_ff=19)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    a,b,report=build_matched_models(cfg,device=device)
    assert report["bijection"] and report["count"] == len(dict(a.named_parameters()))
    p=torch.tensor([1,2,3,4],device=device); c=torch.tensor([5,6],device=device)
    assert check_equivalence(a,b,p,c,atol=1e-5,rtol=1e-4)["allclose"]
    assert check_equivalence(a,b,torch.tensor([1],device=device),c[:1],atol=1e-5,rtol=1e-4)["allclose"]


def test_cache_bytes_unequal_dimensions():
    k=torch.zeros((3,2),dtype=torch.float32); v=torch.zeros((3,5),dtype=torch.float32)
    assert cache_nbytes([[(k,v)]]) == 3*(2+5)*4


def test_benchmark_reports_n0_n1_and_capacity():
    cfg=BenchmarkConfig(model=ModelConfig(vocab_size=11,max_seq_len=5,d_model=8,num_layers=2,num_heads=2,d_k=2,d_v=3,d_ff=13),warmup=0,repeats=1,prompt_lengths=(3,),generated_tokens=5,generation_lengths=(0,1,5),generation_prompt_length=4,device="cuda:0" if torch.cuda.is_available() else "cpu")
    result=run_benchmark(cfg)
    by={(c["prompt_length"],c["requested_tokens"]):c for c in result["cases"]}
    assert by[(4,0)]["actual_tokens"]==0 and by[(4,0)]["final_cache_length"] is None
    if cfg.device!="cpu":
        assert by[(4,0)]["memory"]["cached"]["pre_model_allocated_mib"] is not None
        assert by[(4,0)]["memory"]["cached"]["baseline_allocated_mib"]>=by[(4,0)]["memory"]["cached"]["pre_model_allocated_mib"]
    assert by[(4,1)]["decode_steps"]==0
    assert by[(3,5)]["actual_tokens"]==2 and by[(3,5)]["decode_steps"]==1
    assert all(c["generation_ids_agree"] for c in result["cases"])
    assert by[(4,1)]["samples"]["cached_decode_total_ms"] is None


def test_known_quantiles_and_phase_call_accounting(monkeypatch):
    assert _quant([1.,2.,3.,4.]) == {"median":2.5,"p25":1.75,"p75":3.25}
    cfg=ModelConfig(vocab_size=13,max_seq_len=12,d_model=8,num_layers=2,num_heads=2,d_k=2,d_v=3,d_ff=11)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a,b,_=build_matched_models(cfg,device=device)
    ids=torch.tensor([1,2,3],device=device); cont=torch.tensor([4,5],device=device)
    calls={"a":0,"bp":0,"bd":0,"grad":[],"uncached_ids":[],"cached_ids":[],"cache_lengths":[]}
    timer_start_prefills=[]
    import llm_foundations.experiments.kv_cache_benchmark as bench
    old_perf=bench.time.perf_counter
    def measured_perf():
        if len(perf_calls)%2==0:timer_start_prefills.append(calls["bp"])
        perf_calls.append(None)
        return old_perf()
    perf_calls=[]
    monkeypatch.setattr(bench.time,"perf_counter",measured_perf)
    olda,oldp,oldd=a.forward,b.prefill,b.decode
    def af(x): calls["a"]+=1;calls["uncached_ids"].append(x.tolist());calls["grad"].append(torch.is_grad_enabled());return olda(x)
    def bp(x): calls["bp"]+=1;calls["grad"].append(torch.is_grad_enabled());return oldp(x)
    def bd(x,c): calls["bd"]+=1;calls["cached_ids"].append(x.tolist());calls["cache_lengths"].append(c[0][0][0].shape[0]);calls["grad"].append(torch.is_grad_enabled());return oldd(x,c)
    a.forward=af;b.prefill=bp;b.decode=bd
    samples=_phase_samples(a,b,ids,cont,n=3,repeats=1,warmup=0,device=device)
    assert calls["a"]==3 and calls["bp"]==2 and calls["bd"]==2
    assert timer_start_prefills==[0,0,0,2]
    assert calls["uncached_ids"]==[[1,2,3],[1,2,3,4],[1,2,3,4,5]]
    assert calls["cached_ids"]==[[4],[5]] and calls["cache_lengths"]==[3,4]
    assert calls["grad"] and not any(calls["grad"])
    assert a.training and b.training
    assert samples["uncached_decode_total_ms"] and samples["cached_decode_total_ms"]


def test_zero_work_calls_no_model_and_wrappers_are_really_called(monkeypatch):
    import llm_foundations.experiments.kv_cache_benchmark as bench
    cfg=BenchmarkConfig(model=ModelConfig(vocab_size=9,max_seq_len=6,d_model=8,num_layers=2,num_heads=2,d_k=2,d_v=3,d_ff=10),warmup=0,repeats=1,prompt_lengths=(2,),generated_tokens=0,generation_lengths=(0,),generation_prompt_length=2,device="cuda:0" if torch.cuda.is_available() else "cpu")
    calls={"forward":0,"prefill":0,"uncached_wrapper":0,"cached_wrapper":0}
    old_forward,old_prefill=bench.DecoderOnlyLM.forward,bench.CachedDecoderOnlyLM.prefill
    def forward(self,ids):calls["forward"]+=1;return old_forward(self,ids)
    def prefill(self,ids):calls["prefill"]+=1;return old_prefill(self,ids)
    monkeypatch.setattr(bench.DecoderOnlyLM,"forward",forward)
    monkeypatch.setattr(bench.CachedDecoderOnlyLM,"prefill",prefill)
    # Kiểm tra N=0 không gọi model rồi xác nhận wrapper công khai thực sự được gọi
    zero=run_benchmark(cfg)
    assert calls["forward"]==0 and calls["prefill"]==0
    original_u,original_c=bench.generate,bench.generation
    def wrapped_u(*args,**kwargs):calls["uncached_wrapper"]+=1;return original_u(*args,**kwargs)
    def wrapped_c(*args,**kwargs):calls["cached_wrapper"]+=1;return original_c(*args,**kwargs)
    monkeypatch.setattr(bench,"generate",wrapped_u);monkeypatch.setattr(bench,"generation",wrapped_c)
    cfg=BenchmarkConfig(model=cfg.model,warmup=0,repeats=1,prompt_lengths=(2,),generated_tokens=1,generation_lengths=(1,2),generation_prompt_length=2,device=cfg.device)
    generated=run_benchmark(cfg)
    assert calls["uncached_wrapper"]>0 and calls["cached_wrapper"]>0
    budget_cases=[c for c in generated["cases"] if c["sweep"]=="budget"]
    assert len({tuple(c["prompt_ids"]) for c in budget_cases})==1
    longest=max((c["continuation_ids"] for c in budget_cases),key=len)
    assert all(c["continuation_ids"]==longest[:len(c["continuation_ids"])] for c in budget_cases)


def test_source_provenance_is_independent_of_working_directory(monkeypatch,tmp_path):
    import llm_foundations.experiments.kv_cache_benchmark as bench
    from_root=bench._source_metadata()
    monkeypatch.chdir(tmp_path)
    elsewhere=bench._source_metadata()
    assert elsewhere==from_root
    assert "llm_foundations/experiments/kv_cache_benchmark.py" in elsewhere["sha256"]
    assert "llm_foundations/experiments/kv_cache_benchmark.ipynb" in elsewhere["sha256"]


def test_wrapper_exceptions_propagate(monkeypatch):
    import pytest
    import llm_foundations.experiments.kv_cache_benchmark as bench
    def fail(*args,**kwargs):raise RuntimeError("wrapper failed")
    monkeypatch.setattr(bench,"generate",fail)
    with pytest.raises(RuntimeError,match="wrapper failed"):
        bench._call_arm(object(),torch.tensor([1]),1,False)


def test_fresh_artifacts_and_all_plot_exports(tmp_path):
    cfg=BenchmarkConfig(model=ModelConfig(vocab_size=9,max_seq_len=8,d_model=8,num_layers=2,num_heads=2,d_k=2,d_v=3,d_ff=10),warmup=0,repeats=1,prompt_lengths=(2,),generated_tokens=2,generation_lengths=(1,2),generation_prompt_length=2,device="cuda:0" if torch.cuda.is_available() else "cpu")
    result=run_benchmark(cfg);one=save_results(result,tmp_path);two=save_results(result,tmp_path)
    assert one!=two and (one/"raw.csv").exists() and (one/"summary.csv").exists()
    figs=plot_results(result,output_dir=one/"graphs")
    assert len(figs)>=12 and len(list((one/"graphs").glob("*.svg")))==len(figs)
