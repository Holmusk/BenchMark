import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def resolve_device(device_str):
    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cpu")

def load_model_and_tokenizer(cfg, device_str="auto"):
    task = cfg.get("task", "feature_extraction")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])

    if task == "text_classification":
        model = AutoModelForSequenceClassification.from_pretrained(cfg["name"])
    else:
        model = AutoModel.from_pretrained(cfg["name"])

    device = resolve_device(device_str)
    model.to(device)
    model.eval()
    return model, tokenizer, device
