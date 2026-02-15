from typing import Optional

import torch

from .base import BaseProbabilityModel
from .hf_model import HuggingFaceCausalModel


MODEL_CONFIGS = {
    "gpt2": {
        "hf_name": "gpt2",
        "display_name": "GPT-2 (124M)",
        "family": "GPT-2",
        "parameters": "124M",
        "torch_dtype": None,
        "trust_remote_code": False,
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "display_name": "GPT-2 Medium (355M)",
        "family": "GPT-2",
        "parameters": "355M",
        "torch_dtype": None,
        "trust_remote_code": False,
    },
    "opt-1.3b": {
        "hf_name": "facebook/opt-1.3b",
        "display_name": "OPT-1.3B (Meta)",
        "family": "OPT",
        "parameters": "1.3B",
        "torch_dtype": torch.float16,
        "trust_remote_code": False,
    },
    "phi-2": {
        "hf_name": "microsoft/phi-2",
        "display_name": "Phi-2 (2.7B, Microsoft)",
        "family": "Phi",
        "parameters": "2.7B",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    },
    "mistral-7b": {
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "display_name": "Mistral-7B (Mistral AI)",
        "family": "Mistral",
        "parameters": "7B",
        "torch_dtype": torch.float16,
        "trust_remote_code": False,
    },
}


def create_model(
    model_key: str,
    device: Optional[str] = None,
) -> BaseProbabilityModel:
    if model_key not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(
            f"Неизвестная модель: '{model_key}'. "
            f"Доступные модели: {available}"
        )

    config = MODEL_CONFIGS[model_key]

    model = HuggingFaceCausalModel(
        model_name=config["hf_name"],
        device=device,
        torch_dtype=config.get("torch_dtype"),
        trust_remote_code=config.get("trust_remote_code", False),
    )

    return model


def list_available_models() -> list[dict]:
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "key": key,
            "display_name": config["display_name"],
            "hf_name": config["hf_name"],
            "family": config["family"],
            "parameters": config["parameters"],
        })
    return models


def get_model_display_name(model_key: str) -> str:
    if model_key in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_key]["display_name"]
    return model_key
