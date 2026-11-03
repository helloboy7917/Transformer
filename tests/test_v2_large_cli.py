import sys
import types
import subprocess
import os
from korean_transformer_v2.checkpoint import read_checkpoint
from korean_transformer_v2.__main__ import parser
from korean_transformer_v2.data_loader import stream_huggingface


def test_recommended_defaults_and_unlimited_option():
    args = parser().parse_args(["train", "--dataset", "kormix", "--output", "checkpoints/new.pt", "--all-documents"])
    assert (args.embedding_dim, args.num_layers, args.num_heads, args.context_length) == (256, 6, 8, 512)
    assert (args.vocab_size, args.batch_size, args.gradient_accumulation_steps) == (8000, 4, 8)
    assert args.all_documents and args.max_steps is None


def test_dataset_presets_stream_without_materializing(monkeypatch):
    calls = []
    def load(repo, **kwargs):
        calls.append((repo, kwargs))
        return iter([{"text": "한국어"}])
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load))
    assert list(stream_huggingface("korean-wikipedia", revision="pinned")) == ["한국어"]
    assert calls[0][0] == "wikimedia/wikipedia"
    assert calls[0][1]["name"] == "20231101.ko"
    assert calls[0][1]["streaming"] is True
    assert list(stream_huggingface("kormix")) == ["한국어"]
    assert calls[1][0] == "AdaMLLab/KorMix"


def test_cli_extends_target_and_exports_tokenizer(tmp_path):
    data = tmp_path / "input.txt"
    data.write_text("\n".join(f"한국어 문서 번호 {i}" for i in range(100)), encoding="utf-8")
    path, tokenizer = tmp_path / "model.pt", tmp_path / "v2.tokenizer.json"
    command = [sys.executable, "-m", "korean_transformer_v2", "train", "--data", str(data),
               "--output", str(path), "--cache-dir", str(tmp_path / "cache"), "--device", "cpu"]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    settings = ["--max-steps", "1", "--warmup-steps", "0", "--context-length", "4",
                "--embedding-dim", "8", "--num-heads", "2", "--num-layers", "1",
                "--batch-size", "2", "--gradient-accumulation-steps", "1", "--eval-batches", "1"]
    subprocess.run(command + settings + ["--export-tokenizer", str(tokenizer)], env=env, capture_output=True, check=True)
    before = read_checkpoint(path)
    subprocess.run(command + ["--resume", str(path), "--max-steps", "2", "--tokenizer-file", str(tokenizer)],
                   env=env, capture_output=True, check=True)
    after = read_checkpoint(path)
    assert after["step"] == after["config"]["max_steps"] == 2
    assert before["tokenizer"] == after["tokenizer"]
