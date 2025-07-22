import os
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
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
    model_path = cfg["name"]
    task = cfg.get("task", "feature_extraction")
    model_onnx_path = os.path.join(model_path, "model.onnx")

    if os.path.exists(model_onnx_path):
        # ONNX model loading
        from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if task == "text_classification":
            model = ORTModelForSequenceClassification.from_pretrained(model_path)
        else:
            model = ORTModelForFeatureExtraction.from_pretrained(model_path)
        device = resolve_device(device_str)
        model.to(device)
        return model, tokenizer, device
    else:
        # HuggingFace model loading (existing logic)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if task == "text_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            model = AutoModel.from_pretrained(model_path)
        device = resolve_device(device_str)
        model.to(device)
        model.eval()
        return model, tokenizer, device
