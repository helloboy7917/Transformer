import pytest
import torch

from services.dataset_loader import DatasetSpec, load_dataset_bundle
from services.model_manager import ModelManager
from utils.checkpoint import read_checkpoint


def write_corpus(path, prefix="abc"):
    documents = [
        f"{prefix} pattern number {number} repeats with a short ending."
        for number in range(8)
    ]
    path.write_text("\n\n".join(documents), encoding="utf-8")
    return path


def small_config():
    return {
        "embedding_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "context_length": 8,
        "feedforward_dim": 32,
        "dropout": 0.0,
        "batch_size": 2,
        "learning_rate": 0.01,
        "max_steps": 2,
        "eval_interval": 1,
        "eval_batches": 1,
        "checkpoint_interval": 1,
        "validation_fraction": 0.25,
        "seed": 17,
    }


def local_spec(path, **overrides):
    values = {
        "source_type": "local",
        "data_path": str(path),
        "max_samples": None,
        "max_chars": None,
        "validation_fraction": 0.25,
        "seed": 17,
    }
    values.update(overrides)
    return DatasetSpec(**values)


def test_models_are_independent_and_training_resumes(tmp_path):
    manager = ModelManager(models_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    corpus = write_corpus(tmp_path / "first.txt")
    created = manager.create("model-a", local_spec(corpus), small_config())
    assert created["status"] == "initialized"
    assert created["trained_steps"] == 0
    model_dir = manager.model_dir("model-a")
    for filename in ("config.json", "dataset_info.json", "tokenizer.json", "initial.pt"):
        assert (model_dir / filename).is_file()
    assert not (model_dir / "best.pt").exists()

    with pytest.raises(FileExistsError, match="already exists"):
        manager.create("model-a", local_spec(corpus), small_config())

    first_train = manager.train("model-a", max_steps=2, device="cpu")
    assert first_train["status"] == "trained"
    assert first_train["trained_steps"] == 2
    assert (model_dir / "best.pt").is_file()
    assert (model_dir / "last.pt").is_file()
    assert (model_dir / "training_log.csv").is_file()

    resumed = manager.train("model-a", max_steps=3, device="cpu")
    assert resumed["trained_steps"] == 3
    assert len(resumed["training_log_tail"]) >= 3
    source_last_before = read_checkpoint(model_dir / "last.pt", map_location="cpu")

    cloned = manager.clone("model-a", "model-clone")
    clone_dir = manager.model_dir("model-clone")
    assert cloned["status"] == "cloned"
    assert cloned["tokenizer_vocabulary_size"] == resumed["tokenizer_vocabulary_size"]
    clone_last = read_checkpoint(clone_dir / "last.pt", map_location="cpu")
    assert clone_last["step"] == source_last_before["step"]
    assert clone_last["model_state_dict"].keys() == source_last_before["model_state_dict"].keys()
    for key in clone_last["model_state_dict"]:
        torch.testing.assert_close(
            clone_last["model_state_dict"][key], source_last_before["model_state_dict"][key]
        )

    after_clone_training = manager.train("model-clone", max_steps=4, device="cpu")
    assert after_clone_training["trained_steps"] == 4
    source_last_after = read_checkpoint(model_dir / "last.pt", map_location="cpu")
    assert source_last_after["step"] == source_last_before["step"] == 3

    generated = manager.generate("model-a", "a", max_new_tokens=4, greedy=True, device="cpu")
    assert len(generated) == 5
    rows = {row["name"]: row for row in manager.list_models()}
    assert rows["model-a"]["status"] == "trained"
    assert rows["model-a"]["last_step"] == 3
    assert rows["model-clone"]["last_step"] == 4
    info = manager.info("model-a")
    assert info["dataset"]["source_type"] == "local"
    assert info["config"]["context_length"] == 8


def test_vocabulary_is_built_per_model_and_incompatible_finetune_is_rejected(tmp_path):
    manager = ModelManager(models_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    first = write_corpus(tmp_path / "first.txt", prefix="abc")
    second = write_corpus(tmp_path / "second.txt", prefix="XYZ🧪")
    first_model = manager.create("first", local_spec(first), small_config())
    second_model = manager.create("second", local_spec(second), small_config())
    assert first_model["tokenizer_vocabulary_size"] != second_model["tokenizer_vocabulary_size"]
    first_initial = read_checkpoint(manager.model_dir("first") / "initial.pt", map_location="cpu")
    second_initial = read_checkpoint(manager.model_dir("second") / "initial.pt", map_location="cpu")
    assert not torch.equal(
        first_initial["model_state_dict"]["blocks.0.attention.qkv_projection.weight"],
        second_initial["model_state_dict"]["blocks.0.attention.qkv_projection.weight"],
    )

    incompatible = tmp_path / "incompatible.txt"
    incompatible.write_text(
        "solar ☀ sentence one.\n\nsolar ☀ sentence two.\n\nsolar ☀ sentence three.\n\n"
        "solar ☀ sentence four.\n",
        encoding="utf-8",
    )
    old_fingerprint = manager.info("first")["dataset"]["dataset_fingerprint"]
    with pytest.raises(RuntimeError, match="outside its tokenizer vocabulary"):
        manager.train("first", dataset_override=local_spec(incompatible), device="cpu")
    assert manager.info("first")["dataset"]["dataset_fingerprint"] == old_fingerprint


def test_finetune_dataset_is_recorded_only_after_success_and_can_resume(tmp_path, monkeypatch):
    import services.model_manager as model_manager_module

    manager = ModelManager(models_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    original = write_corpus(tmp_path / "original.txt")
    manager.create("recoverable", local_spec(original), small_config())
    original_fingerprint = manager.info("recoverable")["dataset"]["dataset_fingerprint"]

    finetune = tmp_path / "finetune.txt"
    original_training_documents = load_dataset_bundle(local_spec(original)).train_documents
    finetune.write_text(
        "\n\n".join(document[::-1] for document in original_training_documents),
        encoding="utf-8",
    )

    def fail_before_training(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(model_manager_module, "train_loop", fail_before_training)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        manager.train("recoverable", dataset_override=local_spec(finetune), device="cpu")
    assert manager.info("recoverable")["dataset"]["dataset_fingerprint"] == original_fingerprint

    monkeypatch.undo()
    trained = manager.train(
        "recoverable", max_steps=1, dataset_override=local_spec(finetune), device="cpu"
    )
    assert trained["trained_steps"] == 1
    updated_dataset = manager.info("recoverable")["dataset"]
    assert updated_dataset["dataset_fingerprint"] != original_fingerprint
    assert len(updated_dataset["fine_tuning_datasets"]) == 1

    resumed = manager.train("recoverable", max_steps=2, device="cpu")
    assert resumed["trained_steps"] == 2


def test_manage_cli_creates_and_lists_local_model(tmp_path, capsys):
    from manage import main

    corpus = write_corpus(tmp_path / "cli.txt")
    result = main(
        [
            "create",
            "--models-dir",
            str(tmp_path / "models"),
            "--name",
            "cli-model",
            "--data",
            str(corpus),
            "--embedding-dim",
            "16",
            "--num-heads",
            "4",
            "--num-layers",
            "1",
            "--context-length",
            "8",
            "--dropout",
            "0",
        ]
    )
    assert result == 0
    assert "Created model 'cli-model'" in capsys.readouterr().out
    result = main(["list", "--models-dir", str(tmp_path / "models")])
    assert result == 0
    assert "cli-model" in capsys.readouterr().out
