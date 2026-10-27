# Mini Transformer (Character GPT)

A small decoder-only language model built directly with PyTorch tensor operations. It is intended for learning and local experiments; it does not download a pretrained model and does not use Hugging Face Transformers, `nn.Transformer`, or `nn.MultiheadAttention`.

## Project layout

```text
config.py                   Model and training hyperparameters
model/embedding.py          Token and learned position embeddings
model/attention.py          Scaled dot-product and causal multi-head attention
model/feedforward.py        GELU feed-forward network
model/transformer_block.py  Pre-LayerNorm block and residual paths
model/gpt.py                Decoder-only language model and generation
data/tokenizer.py           Character tokenizer
data/dataset.py             Next-character windows and random batches
data/sample.txt             Original, small example corpus
train.py                    Training, evaluation, checkpoints, and resume
generate.py                 Sampling from a saved checkpoint
utils/checkpoint.py         Atomic checkpoint save and restore
tests/                      Shape, masking, training, checkpoint, and generation tests
```

## Model and attention

For token IDs with shape `(batch, time)`, the model adds learned token and position vectors to get `(batch, time, embedding_dim)`. Each of the `num_layers` blocks applies pre-normalized causal self-attention and a position-wise feed-forward network, each with a residual connection. A final LayerNorm and linear projection produce vocabulary logits with shape `(batch, time, vocab_size)`.

For each head, attention computes

```text
scores  = Q @ K.transpose(-2, -1) / sqrt(head_dim)
weights = softmax(scores + causal_mask)
output  = weights @ V
```

The causal mask sets all score positions above the diagonal to the dtype's minimum value before softmax. Thus position `t` can use positions `0..t`, but cannot use any later token. The heads are split from one QKV projection, computed in parallel, joined along the channel dimension, and passed through an output projection.

Training uses next-character targets shifted one position to the right, cross-entropy loss, backpropagation, AdamW, and gradient clipping. Validation loss is measured periodically. The default model has 128 embedding channels, 4 heads, 4 blocks, a 128-token context, a 512-channel feed-forward layer, and 0.1 dropout. The vocabulary comes from the corpus, so the exact parameter count depends on the text:

```text
parameters = 809,728 + 257 * vocabulary_size
```

This formula applies to the default dimensions and untied token/output embeddings. The trainer prints the exact count for the selected corpus and configuration.

## Install and train

From this directory, install the dependencies and start training:

```bash
pip install -r requirements.txt
python train.py
```

`data/sample.txt` is included so the default command has training data. Use another UTF-8 text file with `--data path/to/corpus.txt`. CUDA is selected automatically when available; use `--device cpu` or `--device cuda` to choose explicitly.

Training reports step, train loss, validation loss, and learning rate. It writes `checkpoints/best.pt`, periodic `step_*.pt` files, and `checkpoints/last.pt`. Resume from the last checkpoint with:

```bash
python train.py --resume checkpoints/last.pt
```

Evaluation rows are also appended to `checkpoints/training_log.csv`.

For a quick run or smaller local experiment, settings can be overridden, for example:

```bash
python train.py --max-steps 20 --batch-size 4 --context-length 64
```

All default model and training values live in `GPTConfig` in `config.py`.

## Generate text

After training, provide a prompt. The generator crops long prompts to the context window and supports greedy decoding, temperature, and top-k sampling:

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --max-new-tokens 100
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --greedy
python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --temperature 0.8 --top-k 20
```

Characters not present in the training vocabulary map to a visible replacement character. Choose prompts using characters present in the training corpus for the most useful output.

## Tests

Run the automated checks with:

```bash
pytest tests/
```

The tests cover tokenizer round trips, embeddings and block shapes, causal masking (including changing future tokens), model logits and gradients, a tiny-corpus overfit check, checkpoint restoration, and text generation.

## Limitations and next steps

Character-level tokens create long sequences and a vocabulary that is tied to one corpus. This compact model has limited capacity and needs a larger, clean dataset for useful language generation. Possible extensions include a separately implemented BPE tokenizer, learning-rate scheduling, mixed precision, batched data loading, weight tying, and tracking loss curves over longer runs.
