import os
import subprocess
import sys


def test_cli_train_resume_and_generate(tmp_path):
    data = tmp_path / "input.txt"
    data.write_text("\n".join(f"한국어 문장 {i} 테스트" for i in range(40)), encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    base = [sys.executable, "-m", "korean_transformer_v2"]
    train = ["train", "--data", str(data), "--output", str(checkpoint), "--device", "cpu"]
    settings = ["--max-steps", "2", "--warmup-steps", "1", "--context-length", "4",
                "--embedding-dim", "8", "--num-heads", "2", "--num-layers", "1",
                "--batch-size", "2", "--eval-batches", "1", "--vocab-size", "260"]
    first = subprocess.run(base + train + settings + ["--stop-after", "1"], env=env, text=True, encoding="utf-8", capture_output=True, check=True)
    assert "step=1" in first.stdout
    resumed = subprocess.run(base + train + ["--resume", str(checkpoint)], env=env, text=True, encoding="utf-8", capture_output=True, check=True)
    assert "step=2" in resumed.stdout
    generated = subprocess.run(base + ["generate", "--checkpoint", str(checkpoint), "--prompt", "안녕", "--max-new-tokens", "0", "--device", "cpu"], env=env, text=True, encoding="utf-8", capture_output=True, check=True)
    assert generated.stdout.strip() == "안녕"
