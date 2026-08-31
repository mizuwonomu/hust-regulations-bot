try:
    import torch
    from sentence_transformers import CrossEncoder
    from torch import nn

    from src.rag.config import RERANKER_MODEL

except Exception as e:  # noqa: BLE001
    CrossEncoder = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

def get_device():
    if torch.cuda.is_available():
        return "cuda"

def load_reranker():
    """Lazily load and return a CrossEncoder reranker"""
    device = get_device()
    print(f"Đang load reranker model '{RERANKER_MODEL}'..... ")
    if CrossEncoder is None:
        raise ImportError(
            "sentence-transformers is required for the reranker"
            "but is not installed. Please install it, e.g: "
            "uv add sentence-transformers\n"
        ) from _IMPORT_ERROR

    model_name = RERANKER_MODEL
    max_length = 512
    model_kwargs = {"dtype": "bfloat16"}

    return CrossEncoder(model_name, 
    max_length=max_length, 
    device=device, 
    model_kwargs=model_kwargs, 
    activation_fn=nn.Sigmoid()
    ) 