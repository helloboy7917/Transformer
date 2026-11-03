# V2 최적화 검증 기록

검증일: 2026-11-03 (Asia/Seoul). 대규모 사전 학습은 실행하지 않았습니다.

## 환경과 자동 테스트

- Windows / 프로젝트 `.venv` Python 3.13 계열 / PyTorch 2.14.1+cu126
- NVIDIA GeForce RTX 4070, VRAM 약 12GiB
- tokenizers 0.23.3
- 전체 테스트 **53 passed** (18.18초)
- `git diff --check` 및 CLI `train --help` 확인

실행 명령:

```powershell
$env:HF_DATASETS_CACHE = Join-Path (Get-Location) 'dataset_cache/hf-tests'
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp=.test-release-v2 tests
```

테스트는 V1 회귀, 기존 Python BPE, 빠른 BPE의 8,000/16,000 목표 및 Unicode 복원, 디스크 중복 제거·분할, 기존 데이터 지문 호환성, SDPA와 기존 가중치 호환, Gradient Accumulation 동등성, Scheduler, 체크포인트/RNG 복원, 목표 Step 확장, 이전 파일 보존, CLI 저장·복원·생성, 네트워크 없는 캐시 재사용, 캐시 길이 오류, BF16/FP16 CUDA 학습·재개를 포함합니다.

처음 전체 테스트에서 샌드박스 밖 HF 기본 캐시 쓰기가 거부되어 프로젝트 안으로 `HF_DATASETS_CACHE`를 지정했습니다. 최종 실행은 전부 통과했습니다.

## 실제 한국어 위키백과 GPU 검증

`wikimedia/wikipedia`, `20231101.ko`, Revision `3e1f92c331f318af862b87e2319ed5dc26d80f5d`의 **200문서만** HF 스트리밍으로 읽었습니다. 추천 설정인 Vocabulary 8,000, Embedding 256, Layers 6, Heads 8, Context 512, Batch 4, 누적 8, LR 0.0003, BF16을 사용했습니다. 검증은 매 Step 한 배치, Warmup은 1 Step입니다.

| 단계 | Train Loss | Validation Loss | 학습 Tokens/sec | 최대 GPU 할당 메모리 |
| --- | --- | --- | --- | --- |
| 첫 Step | 9.0362 | 8.9064 | 37,442 | 약 409MiB |
| 두 번째 Step | 8.8934 | 8.9058 | 109,582 | 약 478MiB |
| 캐시 재개 후 세 번째 Step | 8.8867 | 8.8316 | 52,434 | 약 477MiB |

목표 2 Step을 완료한 파일에서 목표를 3으로 확장하고, 준비 캐시만으로 원본·네트워크 없이 재개했습니다. 생성 명령도 성공했으며 짧은 학습으로 문장 품질을 평가하지 않습니다. 토큰 속도는 짧은 검증 Step의 측정값이며 검증·체크포인트 저장·데이터 준비 시간을 제외합니다. 대규모 학습의 지속 처리량이나 품질을 보장하지 않습니다.

로컬 결과는 Git에서 제외한 `checkpoints/v2-optimization-smoke/wikipedia-200.pt`, `wikipedia-200.metrics.jsonl`, `wikipedia-200.tokenizer.json` 및 `dataset_cache/v2-wiki-smoke/`에 있습니다. 기존 체크포인트 `checkpoints/korean-v2-smoke.pt`도 Step 50 상태로 읽어 가중치 로드와 GPU 생성을 확인했습니다.

별도로 200개 합성 문서에서도 실제 Vocabulary 8,000과 같은 추천 구조로 BF16 학습·목표 확장·생성을 확인했습니다. KorMix는 공식 데이터 카드와 Mock 기반 프리셋/스트리밍 테스트로 확인했으며 실제 원격 KorMix 학습은 실행하지 않았습니다.

## 모듈별 커밋과 원격 확인

각 모듈의 테스트 후 해당 파일만 커밋하고 `origin/feature/korean-transformer-v2`에 Push한 뒤 `git ls-remote`로 반영을 확인했습니다.

| Commit | 내용 |
| --- | --- |
| e1c20b4 | SQLite 중복 제거·분할, 메모리 매핑 토큰 |
| 36f3965 | Rust BPE, Tokenizer 저장·복원, 기존 형식 유지 |
| ddc73f0 | 체크포인트 이전 버전 보존, 목표 Step 확장 |
| ce127cb | 디스크 학습 파이프라인, SDPA, GPU 지표 |
| e12734c | 한국어 데이터셋 프리셋, 추천 CLI 설정 |
| 7849bc1 | 준비 캐시 재개, BF16/FP16 및 캐시 검증 |

README·학습 안내·본 기록은 별도 문서 커밋으로 반영합니다. `main` 변경, 강제 Push, 파괴적 Git 명령은 사용하지 않았습니다. 데이터셋, 모델 가중치, 인증 토큰은 커밋하지 않았습니다.
