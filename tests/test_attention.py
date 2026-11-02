import torch

from model.attention import CausalSelfAttention, scaled_dot_product_attention


def test_scaled_attention_shape_and_weights_are_causal():
    q = torch.randn(2, 3, 5, 4)
    output, weights = scaled_dot_product_attention(q, q, q, causal=True, return_weights=True)
    assert output.shape == q.shape
    assert weights.shape == (2, 3, 5, 5)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0


def test_future_tokens_do_not_change_earlier_attention_outputs():
    torch.manual_seed(3)
    attention = CausalSelfAttention(embedding_dim=16, num_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, 4:, :] = torch.randn_like(changed[:, 4:, :]) * 100
    with torch.inference_mode():
        original_output = attention(x)
        changed_output = attention(changed)
    assert original_output.shape == (1, 6, 16)
    torch.testing.assert_close(original_output[:, :4], changed_output[:, :4])
