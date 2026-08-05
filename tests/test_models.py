import pytest
import torch

from deepfmp_dg.models import MODEL_TYPES, create_model


@pytest.mark.parametrize("model_type", list(MODEL_TYPES.keys()))
def test_forward_shapes(model_type):
    torch.manual_seed(0)
    model = create_model(model_type, emb_dim=8, n_tabular=4, n_align=3)
    b = 2
    seller = torch.randn(b, 8)
    buyer = torch.randn(b, 8)
    tab = torch.randn(b, 4)
    align = torch.randn(b, 3)
    if model_type == "visual":
        logits = model(seller, buyer)
    elif model_type == "visual_tab":
        logits = model(seller, buyer, tab)
    else:
        logits = model(seller, buyer, align, tab)
    assert logits.shape == (b, 2)


def test_softmax_gate_weights_sum_to_one():
    torch.manual_seed(0)
    model = create_model("softmax_gate", emb_dim=8, n_tabular=4, n_align=3)
    align = torch.randn(5, 3)
    w = model.get_gate_weights(align)
    assert w.shape == (5, 3)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(5), atol=1e-5, rtol=1e-5)


def test_create_model_unknown_type():
    with pytest.raises(ValueError):
        create_model("nope")
