# Mini Transformer (문자 단위 GPT)

PyTorch의 기본 텐서 연산으로 직접 구현한 소형 Decoder-only 언어 모델입니다. 학습과 실험을 위한 프로젝트이며, 사전 학습 모델을 내려받지 않습니다. Hugging Face Transformers, `nn.Transformer`, `nn.MultiheadAttention`도 사용하지 않습니다.

## 프로젝트 구성

```text
config.py                   모델 및 학습 하이퍼파라미터
model/embedding.py          토큰 임베딩과 학습 가능한 위치 임베딩
model/attention.py          Scaled Dot-Product 및 Causal Multi-Head Attention
model/feedforward.py        GELU Feed Forward Network
model/transformer_block.py  Pre-LayerNorm 블록과 Residual Connection
model/gpt.py                Decoder-only 언어 모델과 텍스트 생성
data/tokenizer.py           문자 단위 토크나이저
data/dataset.py             다음 문자 예측용 데이터와 배치 생성
data/sample.txt             직접 작성한 소규모 샘플 데이터
train.py                    학습, 검증, 체크포인트 저장 및 재개
generate.py                 체크포인트를 이용한 텍스트 생성
utils/checkpoint.py         체크포인트 저장 및 복원
tests/                      차원, 마스킹, 학습, 체크포인트, 생성 테스트
```

## 모델 구조와 Attention

`(batch, time)` 형태의 토큰 ID에 학습 가능한 토큰 임베딩과 위치 임베딩을 더해 `(batch, time, embedding_dim)` 벡터를 만듭니다. 이후 `num_layers`개의 Transformer 블록을 통과합니다. 각 블록은 Pre-LayerNorm 방식의 Causal Self-Attention과 위치별 Feed Forward Network를 적용하고, 각 계산 결과를 Residual Connection으로 입력에 더합니다. 마지막 LayerNorm과 선형 투영은 `(batch, time, vocab_size)` 형태의 어휘 logits을 출력합니다.

각 Attention Head는 다음 계산을 수행합니다.

```text
scores  = Q @ K.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(scores + causal_mask)
output  = weights @ V
```

Causal Mask는 대각선 위의 Attention Score를 softmax 전에 해당 자료형의 최솟값으로 바꿉니다. 따라서 위치 `t`는 `0..t`까지의 토큰만 참고하고 이후 토큰은 볼 수 없습니다. Q, K, V는 하나의 선형 투영에서 나뉘며, 여러 Head를 병렬 계산한 뒤 연결하고 출력 투영을 거칩니다.

학습에서는 한 위치 오른쪽으로 이동한 다음 문자들을 정답으로 사용합니다. Cross Entropy Loss를 계산하고 역전파와 AdamW로 가중치를 갱신하며, Gradient Clipping도 적용합니다. 검증 손실은 일정 간격으로 계산합니다. 기본 모델 설정은 임베딩 차원 128, Head 4개, 블록 4개, 문맥 길이 128, Feed Forward 차원 512, Dropout 0.1입니다. 어휘 크기는 학습 데이터에 따라 달라지므로 파라미터 수도 달라집니다.

```text
파라미터 수 = 809,728 + 257 × 어휘 크기
```

위 식은 기본 차원과 토큰 임베딩·출력 임베딩을 공유하지 않는 설정에 해당합니다. 실제 파라미터 수는 선택한 데이터와 설정을 기준으로 학습 스크립트가 출력합니다.

## 설치 및 학습

프로젝트 폴더에서 의존성을 설치하고 학습을 시작합니다.

```bash
pip install -r requirements.txt
python train.py
```

기본 실행을 위한 `data/sample.txt`를 포함했습니다. 다른 UTF-8 텍스트를 사용하려면 `--data path/to/corpus.txt`를 지정합니다. CUDA를 사용할 수 있으면 자동으로 선택하며, `--device cpu` 또는 `--device cuda`로 실행 장치를 지정할 수도 있습니다.

학습 중 Step, 학습 손실, 검증 손실, 학습률을 표시합니다. `checkpoints/best.pt`, 주기별 `step_*.pt`, `checkpoints/last.pt`를 저장하고 손실 기록은 `checkpoints/training_log.csv`에 추가합니다. 마지막 체크포인트에서 학습을 재개하려면 다음 명령을 사용합니다.

```bash
python train.py --resume checkpoints/last.pt
```

빠른 실행이나 작은 실험에서는 설정을 바꿀 수 있습니다.

```bash
python train.py --max-steps 20 --batch-size 4 --context-length 64
```

기본 모델 및 학습 설정은 `config.py`의 `GPTConfig`에서 관리합니다.

## 텍스트 생성

학습 후 Prompt를 지정해 텍스트를 생성합니다. Prompt가 문맥 길이보다 길면 최근 토큰만 사용합니다. Greedy Decoding, Temperature, Top-k Sampling을 지원합니다.

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --max-new-tokens 100
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --greedy
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --temperature 0.8 --top-k 20
```

학습 Vocabulary에 없는 문자는 대체 문자로 표시됩니다. 학습 데이터에 포함된 문자를 Prompt에 사용하면 더 적절한 결과를 얻을 수 있습니다.

## 테스트

자동 테스트를 실행합니다.

```bash
pytest tests/
```

토크나이저 인코딩·디코딩, 임베딩과 블록의 차원, Causal Mask와 미래 토큰 차단, 모델 logits와 gradient, 소규모 데이터 과적합, 체크포인트 복원, 텍스트 생성을 검사합니다.

## 한계와 개선 방향

문자 단위 토큰화는 시퀀스가 길어지고 어휘가 학습 데이터에 종속됩니다. 이 소형 모델은 표현 능력이 제한적이므로 유용한 문장을 생성하려면 더 크고 정제된 데이터가 필요합니다. 향후에는 BPE 토크나이저 직접 구현, 학습률 스케줄링, Mixed Precision, 배치 데이터 로딩, 가중치 공유, 장기 학습 손실 곡선 기록 등을 추가할 수 있습니다.