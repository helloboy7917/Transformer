import sys
import types

from korean_transformer_v2.checkpoint import read_checkpoint
from korean_transformer_v2.data_loader import stream_huggingface
from korean_transformer_v2.inference import generate_text
from korean_transformer_v2.model import V2Config, build_model
from korean_transformer_v2.pipeline import train_documents
from korean_transformer_v2.tokenizer import KoreanBPETokenizer


def test_stream_train_reload_and_generate(tmp_path, monkeypatch):
    records = [{"text": f"한국어 테스트 문서 번호 {i}"} for i in range(50)]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: iter(records)))
    source = stream_huggingface("example/corpus", max_documents=50)
    config = V2Config(embedding_dim=8, num_heads=2, num_layers=1, context_length=4,
                      batch_size=2, max_steps=1, warmup_steps=0, eval_batches=1)
    path = tmp_path / "model.pt"
    assert train_documents(source, config=config, output_path=path, vocab_size=260)["step"] == 1
    state = read_checkpoint(path)
    tokenizer = KoreanBPETokenizer.from_dict(state["tokenizer"])
    model = build_model(V2Config.from_dict(state["config"]))
    model.load_state_dict(state["model"])
    text = generate_text(model, tokenizer, "안녕", max_new_tokens=4, greedy=True)
    assert text.startswith("안녕")
