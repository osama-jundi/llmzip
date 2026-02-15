import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llmzip import Compressor, Decompressor, ContextStrategy, list_available_models
from llmzip.arithmetic import ArithmeticEncoder, ArithmeticDecoder
from llmzip.context import ContextManager, ContextStrategy
from llmzip.file_format import FileMetadata, save_compressed, load_compressed
from llmzip.preprocessing import Preprocessor


def test_list_models():
    print("\n" + "=" * 60)
    print("ТЕСТ: Список доступных моделей")
    print("=" * 60)

    models = list_available_models()
    assert len(models) == 5, f"Ожидалось 5 моделей, получено {len(models)}"

    for m in models:
        print(f"  {m['key']:15s} | {m['display_name']:30s} | {m['family']}")

    print("✓ Пройден")


def test_arithmetic_coding():
    print("\n" + "=" * 60)
    print("ТЕСТ: Арифметическое кодирование")
    print("=" * 60)

    vocab_size = 100
    num_symbols = 50
    np.random.seed(42)

    symbols = np.random.randint(0, vocab_size, size=num_symbols)

    all_probs = []
    for _ in range(num_symbols):
        p = np.random.dirichlet(np.ones(vocab_size))
        all_probs.append(p.astype(np.float64))

    encoder = ArithmeticEncoder()
    encoder.create_encoder()

    for i in range(num_symbols - 1, -1, -1):
        probs = all_probs[i].astype(np.float32)
        probs /= probs.sum()
        encoder.encode_symbol(int(symbols[i]), probs)

    compressed = encoder.finish()
    print(f"  Символов: {num_symbols}")
    print(f"  Сжатый размер: {len(compressed) * 4} байт")

    decoder = ArithmeticDecoder(compressed)
    decoded = []
    for i in range(num_symbols):
        probs = all_probs[i].astype(np.float32)
        probs /= probs.sum()
        sym = decoder.decode_symbol(probs)
        decoded.append(sym)

    assert list(symbols) == decoded, "Декодированные символы не совпадают!"
    print("✓ Пройден")


def test_preprocessing():
    print("\n" + "=" * 60)
    print("ТЕСТ: Предобработка")
    print("=" * 60)

    preprocessor = Preprocessor()

    test_cases = [
        "Hello, World!",
        "Line1\nLine2\nLine3",
        "Mixed\r\nline\rendings\n",
        "Привет, мир! Unicode тест 🎉",
        "",
    ]

    for text in test_cases:
        processed = preprocessor.preprocess(text)
        restored = preprocessor.reverse_preprocess(processed)
        expected = text.replace("\r\n", "\n").replace("\r", "\n")
        assert restored == expected, f"Не совпадает: {text!r}"
        print(f"  '{text[:30]}...' → OK")

    print("✓ Пройден")


def test_context_manager():
    print("\n" + "=" * 60)
    print("ТЕСТ: Контекстное окно")
    print("=" * 60)

    tokens = list(range(20))

    cm = ContextManager(max_context_length=5, strategy=ContextStrategy.SLIDING_WINDOW)

    ctx = cm.get_context(tokens, 0)
    assert ctx == [], f"Ожидался [], получено {ctx}"

    ctx = cm.get_context(tokens, 3)
    assert ctx == [0, 1, 2], f"Ожидалось [0,1,2], получено {ctx}"

    ctx = cm.get_context(tokens, 10)
    assert ctx == [5, 6, 7, 8, 9], f"Ожидалось [5,6,7,8,9], получено {ctx}"

    print("  Скользящее окно: OK")

    cm_block = ContextManager(max_context_length=5, strategy=ContextStrategy.BLOCK)

    ctx = cm_block.get_context(tokens, 7)
    assert ctx == [5, 6], f"Ожидалось [5,6], получено {ctx}"

    print("  Блочная стратегия: OK")
    print("✓ Пройден")


def test_file_format():
    print("\n" + "=" * 60)
    print("ТЕСТ: Формат файла .llmz")
    print("=" * 60)

    metadata = FileMetadata(
        model_key="gpt2",
        model_name="gpt2",
        vocab_size=50257,
        max_context_length=1024,
        context_strategy="sliding_window",
        encoding="utf-8",
        num_tokens=100,
        original_size=500,
        original_hash="abc123",
    )

    compressed_data = np.array([1, 2, 3, 4, 5], dtype=np.uint32)

    temp_path = os.path.join(tempfile.gettempdir(), "test_llmzip_full.llmz")

    try:
        save_compressed(temp_path, metadata, compressed_data)
        print(f"  Записано: {Path(temp_path).stat().st_size} байт")

        loaded_meta, loaded_data = load_compressed(temp_path)

        assert loaded_meta.model_key == "gpt2"
        assert loaded_meta.vocab_size == 50257
        assert loaded_meta.original_hash == "abc123"
        assert list(loaded_data) == [1, 2, 3, 4, 5]

        print("  Чтение/запись: OK")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("✓ Пройден")


def test_compression_lossless(model_key: str = "gpt2"):
    print("\n" + "=" * 60)
    print(f"ТЕСТ: Сжатие без потерь (модель: {model_key})")
    print("=" * 60)

    test_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In information theory, entropy measures the average uncertainty "
        "of a random variable. Shannon showed that the entropy of a source "
        "determines the theoretical limit of lossless compression. "
        "Large language models can serve as powerful probability estimators "
        "for arithmetic coding, potentially achieving compression ratios "
        "that surpass traditional methods like GZIP and BZIP2. "
        "This is a test of the LLMZip compression system. "
    ) * 3

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.txt")
        compressed_path = os.path.join(tmpdir, "output.llmz")
        restored_path = os.path.join(tmpdir, "restored.txt")

        with open(input_path, "w") as f:
            f.write(test_text)

        original_size = os.path.getsize(input_path)
        print(f"  Исходный размер: {original_size} байт")

        compressor = Compressor(model_key=model_key)
        comp_results = compressor.compress(input_path, compressed_path, verbose=True)

        compressed_size = os.path.getsize(compressed_path)
        print(f"  Сжатый размер: {compressed_size} байт")
        print(f"  Коэффициент: {comp_results['ratio']:.4f}")

        decompressor = Decompressor()
        decomp_results = decompressor.decompress(compressed_path, restored_path, verbose=True)

        with open(input_path, "r") as f:
            original = f.read()
        with open(restored_path, "r") as f:
            restored = f.read()

        assert original == restored, "Восстановленные данные НЕ совпадают!"
        assert decomp_results["is_lossless"], "Хеши не совпадают!"

        print(f"\n  ✓ LOSSLESS ПОДТВЕРЖДЕНО: исходный == восстановленный")

    print("✓ Пройден")


def run_all_tests():
    print("=" * 60)
    print("LLMZip v2.0 — Полный набор тестов")
    print("=" * 60)

    test_list_models()
    test_arithmetic_coding()
    test_preprocessing()
    test_context_manager()
    test_file_format()

    print("\n⚠️  Следующий тест загружает модель GPT-2 (~500 MB).")
    print("    Для пропуска нажмите Ctrl+C.\n")

    try:
        test_compression_lossless("gpt2")
    except KeyboardInterrupt:
        print("\n  Пропущен по запросу пользователя.")

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
