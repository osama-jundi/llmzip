#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmzip.models.registry import MODEL_CONFIGS


def download_model(key: str, config: dict) -> None:
    hf_name = config["hf_name"]
    print(f"\n{'='*60}")
    print(f"Загрузка: {config['display_name']} ({hf_name})")
    print(f"{'='*60}")

    print(f"  Токенизатор...", end=" ", flush=True)
    AutoTokenizer.from_pretrained(
        hf_name,
        trust_remote_code=config.get("trust_remote_code", False),
    )
    print("OK")

    print(f"  Веса модели...", end=" ", flush=True)
    AutoModelForCausalLM.from_pretrained(
        hf_name,
        trust_remote_code=config.get("trust_remote_code", False),
    )
    print("OK")

    print(f"  {config['display_name']} — загружена и кеширована.")


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка моделей LLM для экспериментов"
    )
    parser.add_argument(
        "models", nargs="*", default=None,
        help="Ключи моделей (по умолчанию: все 5)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Показать список доступных моделей"
    )
    args = parser.parse_args()

    if args.list:
        print("\nДоступные модели:")
        for key, cfg in MODEL_CONFIGS.items():
            print(f"  {key:15s}  {cfg['display_name']:30s}  {cfg['parameters']}")
        return

    keys = args.models if args.models else list(MODEL_CONFIGS.keys())

    for key in keys:
        if key not in MODEL_CONFIGS:
            print(f"ОШИБКА: модель '{key}' не найдена.")
            print(f"Доступные: {', '.join(MODEL_CONFIGS.keys())}")
            sys.exit(1)

    print(f"Моделей к загрузке: {len(keys)}")
    print(f"Модели: {', '.join(keys)}")

    for key in keys:
        download_model(key, MODEL_CONFIGS[key])

    print(f"\n{'='*60}")
    print(f"Все {len(keys)} моделей загружены.")
    print(f"Модели кешируются в ~/.cache/huggingface/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
