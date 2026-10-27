"""Tạo nhãn profiler theo context và bọc tạm các biên thật của model."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator

from torch import nn
from torch.profiler import record_function as _record_function

from llm_foundations.decoding import greedy_token
from llm_foundations.generation import generate
from llm_foundations.kv_cache.generation import generation as cached_generation

_LEVELS = frozenset({"off", "blocks", "ops"})
_ANNOTATION_LEVEL: ContextVar[str] = ContextVar("kv_profile_annotation_level", default="off")
_ACTIVE_MODELS: ContextVar[tuple[int, ...]] = ContextVar("kv_profile_active_models", default=())


def _validate_level(level: str) -> None:
    if level not in _LEVELS:
        raise ValueError("level phải là 'off', 'blocks' hoặc 'ops'")


@contextmanager
def annotation_scope(level: str = "blocks") -> Iterator[None]:
    """Tạm đặt mức nhãn trong context hiện tại rồi khôi phục mức trước đó.

    Args:
        level: Mức 'off', 'blocks' hoặc 'ops' áp dụng trong scope.
    """
    _validate_level(level)
    token = _ANNOTATION_LEVEL.set(level)
    try:
        yield
    finally:
        _ANNOTATION_LEVEL.reset(token)


def current_annotation_level() -> str:
    """Trả mức nhãn đang hoạt động trong context hiện tại."""
    return _ANNOTATION_LEVEL.get()


@contextmanager
def region(name: str, *, level: str = "blocks") -> Iterator[None]:
    """Đặt nhãn profiler tại một biên thật mà không đọc tensor.

    Args:
        name: Nhãn ổn định của vùng tính toán.
        level: Mức tối thiểu 'blocks' hoặc 'ops' để bật nhãn.
    """
    if level not in {"blocks", "ops"}:
        raise ValueError("region level phải là 'blocks' hoặc 'ops'")
    if not isinstance(name, str) or not name:
        raise ValueError("name phải là chuỗi không rỗng")
    current = _ANNOTATION_LEVEL.get()
    enabled = current == "ops" or (current == "blocks" and level == "blocks")
    if enabled:
        with _record_function(name):
            yield
    else:
        yield


def _patch_method(
    stack: ExitStack,
    instance: Any,
    name: str,
    scope_factory: Callable[[], Any],
    *,
    before_call: Callable[[], Any] | None = None,
) -> None:
    original = getattr(instance, name)
    had_instance_value = name in instance.__dict__
    prior_instance_value = instance.__dict__.get(name)

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        scope = scope_factory() if before_call is None else before_call()
        with scope:
            return original(*args, **kwargs)

    setattr(instance, name, wrapped)

    def restore() -> None:
        if had_instance_value:
            setattr(instance, name, prior_instance_value)
        else:
            delattr(instance, name)

    stack.callback(restore)


def _region_factory(name: str, *, level: str = "blocks") -> Callable[[], Any]:
    return lambda: region(name, level=level)


def _module_scopes(model: nn.Module, arm: str, level: str) -> list[tuple[Any, str, str, str]]:
    scopes: list[tuple[Any, str, str, str]] = []
    for path, module in model.named_modules():
        if not path:
            continue
        parts = path.split(".")
        label: str | None = None
        method: str | None = None
        method_name = "forward"
        scope_level = "blocks"
        if path == "input_embedding":
            label = "embedding"
        elif path == "final_norm":
            label = "final_norm"
        elif path == "lm_head":
            label = "lm_head"
        elif len(parts) == 2 and parts[0] == "blocks":
            if arm == "uncached":
                label = f"layer/{parts[1]}"
            else:
                scopes.extend((module, method_name, f"layer/{parts[1]}/{phase}", "blocks")
                              for phase, method_name in (("prefill", "prefill"), ("decode", "decode")))
        elif parts[0] == "blocks" and len(parts) > 1:
            index = parts[1]
            leaf = parts[-1]
            if leaf in {"normalize_input", "normalized_input"}:
                label = "ln1"
            elif leaf == "ffn_norm":
                label = "ln2"
            elif leaf == "ffn":
                label = "ffn"
            elif arm == "uncached" and leaf == "multi_head_attention":
                label = "attention"
            elif arm == "cached" and leaf == "cached_mha":
                scopes.extend((module, method_name, f"attention/{phase}", "blocks")
                              for phase, method_name in (("prefill", "prefill"), ("decode", "decode")))
            elif arm == "uncached" and len(parts) >= 2 and parts[-2] == "heads" and level == "ops":
                label = f"head/{parts[-1]}"
                scope_level = "ops"
            elif arm == "cached" and len(parts) >= 2 and parts[-2] == "cached_heads" and level == "ops":
                head = parts[-1]
                scopes.extend((module, method_name, f"head/{head}/{phase}", "ops")
                              for phase, method_name in (("prefill", "prefill"), ("decode", "decode")))
        if label is not None:
            scopes.append((module, "forward", label, scope_level))
    return scopes


@contextmanager
def instrument_model(model: nn.Module, *, arm: str, level: str = "blocks") -> Iterator[None]:
    """Bọc tạm các phương thức model và khôi phục đúng thuộc tính sau scope.

    Args:
        model: Một instance learner model đang được capture.
        arm: Nhánh 'cached' hoặc 'uncached' được gọi trong scope.
        level: Mức 'blocks' hoặc 'ops' của các nhãn ngữ nghĩa.
    """
    if arm not in {"cached", "uncached"}:
        raise ValueError("arm phải là 'cached' hoặc 'uncached'")
    if level not in {"blocks", "ops"}:
        raise ValueError("level phải là 'blocks' hoặc 'ops'")
    if not isinstance(model, nn.Module):
        raise TypeError("model phải là torch.nn.Module")
    active = _ACTIVE_MODELS.get()
    identity = id(model)
    if identity in active:
        raise RuntimeError("nested instrumentation on the same model is unsupported")
    active_token = _ACTIVE_MODELS.set((*active, identity))
    try:
        with ExitStack() as stack:
            if arm == "cached":
                decode_index = 0
                _patch_method(stack, model, "prefill", _region_factory("prefill"))

                def decode_scope():
                    nonlocal decode_index
                    decode_index += 1
                    return region(f"decode/{decode_index}")

                _patch_method(stack, model, "decode", decode_scope)
                for instance, name, label, scope_level in _module_scopes(model, arm, level):
                    method = label.rsplit("/", 1)[-1]
                    if method in {"prefill", "decode"} and label.startswith("layer/"):
                        base = label.rsplit("/", 1)[0]
                        _patch_method(stack, instance, name, _region_factory(base))
                    elif label.startswith("attention/") and label.count("/") == 1:
                        _patch_method(stack, instance, name, _region_factory("attention"))
                    elif label.startswith("head/") and label.count("/") == 1:
                        for phase_name in ("prefill", "decode"):
                            _patch_method(stack, instance, phase_name, _region_factory(label, level=scope_level))
                    elif label.startswith("head/") and label.count("/") == 2:
                        _patch_method(stack, instance, name, _region_factory(label.rsplit("/", 1)[0], level=scope_level))
                    else:
                        _patch_method(stack, instance, name, _region_factory(label, level=scope_level))
            else:
                forward_index = 0

                def forward_scope():
                    nonlocal forward_index
                    label = "prefill" if forward_index == 0 else f"prefix_forward/{forward_index}"
                    forward_index += 1
                    return region(label)

                _patch_method(stack, model, "forward", forward_scope)
                for instance, name, label, scope_level in _module_scopes(model, arm, level):
                    _patch_method(stack, instance, name, _region_factory(label, level=scope_level))
            yield
    finally:
        _ACTIVE_MODELS.reset(active_token)


def run_instrumented_generation(
    model: nn.Module,
    token_ids,
    *,
    arm: str,
    max_new_tokens: int,
    level: str = "blocks",
):
    """Chạy public greedy wrapper với scope tạm trên các biên model thật.

    Args:
        model: Instance learner model cần đo.
        token_ids: Prompt IDs truyền nguyên trạng vào public wrapper.
        arm: Nhánh 'cached' hoặc 'uncached'.
        max_new_tokens: Budget truyền cho generation wrapper hiện có.
        level: Mức region 'blocks' hoặc 'ops'.
    """
    with annotation_scope(level):
        with instrument_model(model, arm=arm, level=level):
            with region("generation"):
                selector_index = 0

                def selector(logits):
                    nonlocal selector_index
                    index = selector_index
                    selector_index += 1
                    with region(f"selector/{index}"):
                        return greedy_token(logits)

                if arm == "cached":
                    return cached_generation(
                        model, token_ids, max_new_tokens=max_new_tokens, selector=selector,
                    )
                if arm == "uncached":
                    return generate(
                        model, token_ids, max_new_tokens=max_new_tokens, selector=selector,
                    )
                raise ValueError("arm phải là 'cached' hoặc 'uncached'")
