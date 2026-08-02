import torch
import torch.nn as nn

from gradient_methods_nn_regression.model import TinyRegressionModel


def test_tiny_regression_model_layers_have_expected_shapes() -> None:
    model = TinyRegressionModel()

    assert isinstance(model, nn.Module)
    assert model.fc1.in_features == 6
    assert model.fc1.out_features == 16
    assert model.fc2.in_features == 16
    assert model.fc2.out_features == 1


def test_tiny_regression_model_forward_returns_batch_output_shape() -> None:
    model = TinyRegressionModel()
    x = torch.randn(4, 6)

    y = model(x)

    assert y.shape == (4, 1)


def test_tiny_regression_model_is_deterministically_initialized_with_fixed_seed() -> None:
    torch.manual_seed(7)
    model_a = TinyRegressionModel()

    torch.manual_seed(7)
    model_b = TinyRegressionModel()

    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items()
    ):
        assert name_a == name_b
        torch.testing.assert_close(tensor_a, tensor_b)


def test_tiny_regression_model_state_dict_can_be_copied_and_loaded() -> None:
    model_a = TinyRegressionModel()
    model_b = TinyRegressionModel()

    model_b.load_state_dict(model_a.state_dict())

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        torch.testing.assert_close(param_a, param_b)


def test_tiny_regression_model_forward_output_is_finite_for_generated_batch() -> None:
    model = TinyRegressionModel()
    x = torch.randn(8, 6)

    y = model(x)

    assert y.shape == (8, 1)
    assert torch.isfinite(y).all()


def test_tiny_regression_model_counts_trainable_parameters() -> None:
    model = TinyRegressionModel()

    assert model.count_trainable_parameters() == 129
