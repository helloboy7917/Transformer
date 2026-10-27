# Transformer: 다중 모델 언어 모델 실험

PyTorch로 직접 구현한 Decoder-only Transformer(GPT 구조)를 재사용해 여러 개의 독립적인 문자 단위 언어 모델을 만들고 학습하는 프로젝트입니다. 각 모델은 별도의 설정, 가중치, 토크나이저, 데이터 출처와 학습 기록을 가집니다.

Hugging Face의 사전 학습 모델이나 `nn.Transformer`, `nn.MultiheadAttention`은 사용하지 않습니다. 모델 구조는 `model/`의 직접 구현을 사용하며, Hugging Face Hub는 데이터셋을 가져오는 용도로만 사용합니다.

## Transformer 구조

토큰 ID `(batch, time)`에 토큰 임베딩과 학습 가능한 위치 임베딩을 더합니다. 이어서 Pre-LayerNorm, Causal Multi-Head Self-Attention, Residual Connection, GELU Feed Forward Network로 이루어진 Transformer 블록을 통과합니다. 마지막 LayerNorm과 선형 투영이 다음 토큰의 Vocabulary logits을 출력합니다.

Attention은 `softmax(QKᵀ / √dₖ + mask)V`를 직접 계산합니다. Causal Mask가 미래 위치의 점수를 막아 각 위치가 자기 자신과 과거 토큰만 참조하게 합니다. 학습 시 입력보다 한 문자 오른쪽으로 이동한 토큰을 정답으로 사용하고, Cross Entropy Loss와 AdamW, 역전파, Gradient Clipping으로 가중치를 갱신합니다.

기본 `GPTConfig`는 임베딩 128, Head 4개, 블록 4개, 문맥 길이 128, Feed Forward 512, Dropout 0.1입니다. Vocabulary 크기는 각 데이터셋에서 생성합니다. 기본 차원에서 파라미터 수는 다음과 같습니다.

```text
파라미터 수 = 809,728 + 257 × Vocabulary 크기
```

## 설치

Python 3.11 이상에서 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

## 데이터셋

### Hugging Face Hub 데이터셋

Hub의 Dataset Repository ID를 지정합니다. 기본적으로 `train` 분할을 읽고, `validation`, `valid`, `dev` 분할이 있으면 검증에 사용합니다. 공식 검증 분할이 없으면 문서 단위로 중복을 제거한 뒤 설정된 비율과 Seed로 검증 데이터를 나눕니다.

```bash
python manage.py create --name english-model --dataset username/english-dataset
```

필요한 경우 데이터셋 설정, 파일, 분할, 텍스트 컬럼, Revision을 지정할 수 있습니다.

```bash
python manage.py create --name news-model --dataset username/news-dataset \
  --dataset-config default --data-file "data/*.jsonl" \
  --split train --validation-split validation --text-column article \
  --revision 0123456789abcdef --max-samples 5000 --max-chars 1000000
```

텍스트 파일은 `txt`, 구조화 파일은 JSONL, CSV, Parquet 형식을 지원합니다. 문자열 컬럼이 하나면 자동 선택하며, 후보가 여러 개면 `--text-column`으로 학습할 컬럼을 지정합니다. 데이터는 `streaming=True`로 읽고 기본 한도를 분할별 문서 10,000개, 문자 2,000,000개로 제한합니다. `--max-samples`와 `--max-chars`로 한도를 조정할 수 있습니다.

Hub Revision은 학습 시 실제 Commit SHA로 고정해 `dataset_info.json`에 기록합니다. 데이터셋 캐시는 원본 데이터와 별도인 `dataset_cache/`를 사용합니다. Hub의 공개 Dataset은 바로 사용할 수 있고, 비공개 Dataset은 사용자가 직접 인증해야 합니다. 예를 들어 `hf auth login` 또는 이미 설정한 `HF_TOKEN` 환경 변수를 사용합니다. 프로젝트는 인증 토큰을 저장하거나 출력하지 않습니다. [Hugging Face 데이터셋 로딩 안내](https://huggingface.co/docs/datasets/en/loading), [인증 안내](https://huggingface.co/docs/huggingface_hub/package_reference/authentication)

### 로컬 텍스트

UTF-8 `.txt` 파일을 지정합니다. 빈 줄로 문서를 구분하며 문서 단위로 학습·검증을 나눕니다.

```bash
python manage.py create --name custom-model --data data/custom.txt
```

로컬 원본 파일은 이동하거나 삭제하지 않습니다. 파일 경로와 SHA-256 지문을 모델의 Dataset 정보에 저장합니다.

## 모델 생성

`create`는 모델 디렉터리, 설정, 데이터 출처 정보, 모델별 Character Tokenizer와 무작위 초기 가중치 `initial.pt`를 만듭니다. 생성만 수행하며 학습은 시작하지 않습니다. 모델 이름이 이미 있으면 기존 파일을 보존하고 오류를 출력합니다.

```bash
python manage.py create --name model-a --dataset username/dataset-a
python manage.py create --name model-b --dataset username/dataset-b
python manage.py create --name model-c --data data/custom.txt
```

각 모델의 설정은 생성 시 따로 지정할 수 있습니다.

```bash
python manage.py create --name small-model --data data/custom.txt \
  --embedding-dim 128 --num-heads 4 --num-layers 4 \
  --context-length 128 --feedforward-dim 512 --dropout 0.1 \
  --batch-size 16 --learning-rate 0.0003 --max-steps 1000 \
  --weight-decay 0.1 --grad-clip 1.0
```

호환되지 않는 Head 수나 길이 등의 설정은 생성 단계에서 검증합니다.

## 모델 학습과 재개

모델 이름으로 해당 모델의 데이터, Tokenizer, 가중치와 Optimizer 상태를 불러와 학습합니다.

```bash
python manage.py train --name model-a --max-steps 1000
```

`--max-steps`는 추가할 Step 수가 아니라 목표 전체 Step입니다. 실행 로그에는 모델 이름, 장치, 현재 Step/목표 Step, 실제 학습 손실, 검증 손실, 학습률이 표시됩니다. `training_log.csv`, 주기별 Checkpoint, 최고 Validation Loss의 `best.pt`, 재개용 `last.pt`를 모델 폴더에 저장합니다. 학습이 중단되어도 같은 명령으로 마지막 저장 지점부터 이어갈 수 있습니다.

학습할 데이터셋을 바꾸려면 복제 모델에서 데이터 소스를 지정합니다.

```bash
python manage.py train --name model-c --data data/compatible.txt --max-steps 1500
```

Fine-tuning 데이터의 모든 문자는 기존 모델의 Vocabulary에 있어야 합니다. 모르는 문자가 있으면 임의로 `<unk>`에 매핑하지 않고 오류를 출력합니다. 모델 구조 설정은 저장된 Checkpoint와 비교하며, 생성 후 임의로 바꿀 수 없습니다.

## 모델 목록과 정보

```bash
python manage.py list
python manage.py info --name model-a
```

목록에는 모델 이름, 데이터 출처, 상태, 파라미터 수, 마지막 Step, Validation Loss가 표시됩니다. `info`는 설정, Dataset Revision 또는 파일 지문, Tokenizer 크기, Checkpoint 목록, 최근 학습 로그를 보여 줍니다.

## 복제

```bash
python manage.py clone --source model-a --name model-c
```

모델 설정, 가중치, Tokenizer, 데이터 메타데이터와 학습 로그를 별도의 파일로 복사합니다. 복제된 모델을 추가 학습해도 원본 파일은 바뀌지 않습니다. 다른 데이터로 학습할 때도 기존 Vocabulary와의 호환성을 검사합니다.

## 텍스트 생성

```bash
python manage.py generate --name model-a --prompt "Hello" --max-new-tokens 100
python manage.py generate --name model-a --prompt "Hello" --temperature 0.8 --top-k 20
python manage.py generate --name model-a --prompt "Hello" --greedy
```

학습된 Checkpoint와 모델별 Tokenizer를 불러와 생성합니다. 아직 학습하지 않은 모델은 학습이 필요하다는 오류를 출력합니다. Temperature, Top-k, Greedy Decoding, 새 토큰 수를 설정할 수 있습니다.

## 저장 구조와 캐시

```text
trained_models/<model-name>/
├── config.json
├── dataset_info.json
├── tokenizer.json
├── model_state.json
├── initial.pt
├── best.pt
├── last.pt
├── step_*.pt
└── training_log.csv

dataset_cache/
└── Hugging Face Datasets 캐시
```

모델과 캐시 디렉터리는 `.gitignore`에 포함되어 있습니다. 모델 폴더에는 구조 설정, Dataset 출처·Revision, Tokenizer, 가중치, Optimizer 상태, 학습 Step과 손실 기록을 저장합니다. 모델이나 원본 Dataset을 자동으로 삭제하지 않습니다.

## 기존 단일 모델 명령

기존 간단한 실험을 위한 다음 명령도 유지합니다.

```bash
python train.py
python generate.py --prompt "Hello"
pytest tests/
```

다중 모델을 관리할 때는 `manage.py` 명령을 사용합니다.

## 테스트

```bash
python -m pytest -q tests/
```

테스트는 기존 Transformer 구성과 함께 로컬 Dataset 분할, Hub 로더의 Mock 동작과 인증 오류, 모델별 Vocabulary, 중복 이름 방지, 학습·재개, 저장·복원, 목록·정보, 복제 독립성, Tokenizer 호환성, 생성을 검사합니다. Hub에 실제 접속하는 테스트는 자동 실행하지 않습니다.

## 한계

초기 Tokenizer는 Character-level이며 Dataset 문서를 문자열로 다룬 작은 학습용 모델입니다. 고품질 생성에는 더 크고 정제된 데이터와 충분한 학습이 필요합니다. 아직 BPE, 혼합 정밀도, 학습률 스케줄링과 Hugging Face Model Hub 업로드는 지원하지 않습니다.
