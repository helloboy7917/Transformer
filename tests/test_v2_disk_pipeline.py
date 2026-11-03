from dataclasses import replace
from pathlib import Path
import hashlib
import json
import pytest
import torch

from korean_transformer_v2.checkpoint import read_checkpoint
from korean_transformer_v2.corpus import DiskCorpus
from korean_transformer_v2.corpus import load_prepared
from korean_transformer_v2.model import V2Config
from korean_transformer_v2.pipeline import train_documents
from korean_transformer_v2.preprocessing import prepare_documents
from korean_transformer_v2.model import build_model
from model import GPTLanguageModel


def test_disk_fingerprint_matches_legacy(tmp_path):
    docs = [f"한국어 문서 {i}" for i in range(100)]
    corpus = DiskCorpus(docs + docs, tmp_path)
    expected = hashlib.sha256(json.dumps(prepare_documents(docs), ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    assert corpus.fingerprint == expected
    corpus.close()


def test_sdpa_reuses_legacy_weights_and_causal_outputs():
    config = V2Config(embedding_dim=16, num_heads=2, num_layers=1, context_length=8, dropout=0)
    legacy = GPTLanguageModel(config).eval()
    optimized = build_model(config).eval()
    optimized.load_state_dict(legacy.state_dict(), strict=True)
    x = torch.randint(257, (2, 8))
    torch.testing.assert_close(optimized(x), legacy(x), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("device,precision", [("cpu", "auto"), pytest.param("cuda", "bf16", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable")), pytest.param("cuda", "fp16", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_fast_disk_resume_extension_and_metrics(tmp_path, device, precision):
    docs = [f"한국어 위키백과 학습 검증 문서입니다. 번호 {i}" for i in range(100)]
    config = V2Config(embedding_dim=16, num_heads=2, num_layers=1, context_length=8,
                      batch_size=2, max_steps=2, warmup_steps=1, eval_interval=1,
                      eval_batches=1, gradient_accumulation_steps=2, checkpoint_interval=0, precision=precision)
    path = tmp_path / "model.pt"
    kwargs = dict(config=config, output_path=path, device=device, tokenizer_backend="fast", vocab_size=8000)
    result = train_documents(iter(docs), stop_after=1, **kwargs)
    first = read_checkpoint(path)
    assert first["tokenizer"]["type"] == "hf_byte_bpe_v1"
    assert first["training_state"]["tokens_seen"] == 32
    kwargs["config"] = replace(config, max_steps=3)
    def unread_source():
        raise AssertionError("prepared cache must not consume the source")
        yield
    train_documents(unread_source(), resume_path=path, prepared_cache=result["cache"], **kwargs)
    final = read_checkpoint(path)
    assert final["step"] == 3
    assert final["training_state"]["tokens_seen"] == 96
    metrics = [json.loads(line) for line in open(result["metrics"], encoding="utf-8")]
    assert [m["step"] for m in metrics] == [1, 2, 3]
    assert all(m["tokens_per_second"] > 0 and m["validation_loss"] > 0 for m in metrics)
    if device == "cuda":
        assert metrics[-1]["gpu_peak_mb"] > 0
        assert metrics[-1]["precision"] in {"bf16", "fp16"}
    assert list((tmp_path / "model_history").glob("*.pt"))
    manifest_path = Path(result["cache"]) / "prepared.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        load_prepared(result["cache"], config)
