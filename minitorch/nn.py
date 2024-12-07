from typing import Tuple, Optional

from . import operators  # noqa: F401
from .autodiff import Context
from .fast_ops import FastOps  # noqa: F401
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor  # noqa: F401


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Calculate the new height and width after pooling
    new_height = height // kh
    new_width = width // kw

    # Break height and width into chunks of kernel size
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)

    # Permute dimensions to group pooling regions
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()

    # Flatten kernel dimensions
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        output.mean(4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max forward"""
        b = a.f.max_reduce(a, int(dim.item()))
        ctx.save_for_backward(a.f.eq_zip(a, b))
        return b

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward"""
        (a,) = ctx.saved_values
        return a * grad_output, 0.0


def max(input: Tensor, dim: Optional[int]) -> Tensor:
    """Compute the max along a dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to compute max

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        max(output, 4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.

    Args:
    ----
        input: Tensor of any shape.
        rate: Probability of zeroing out each position.
        ignore: If True, ignore dropout and return the input unchanged.

    Returns:
    -------
        Tensor of the same shape as input with dropout applied.

    """
    if ignore or rate <= 0.0:
        # If dropout is ignored or rate is 0, return the input tensor unchanged
        return input

    if rate >= 1.0:
        # If rate is 1.0, zero out all positions
        return input.zeros()

    # Generate a random mask with the same shape as input
    # Apply the mask
    return input * (
        rand(input.shape, backend=input.backend, requires_grad=False) > rate
    )


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: Tensor of size batch x ...
        dim: dimension to apply argmax

    Returns:
    -------
        Tensor of size batch x ...

    """
    return (input == input.max(dim)[0]).float()


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: Tensor of size batch x ...
        dim: dimension to apply softmax

    Returns:
    -------
        Tensor of size batch x ...

    """
    exp_input = input.exp()
    return exp_input / exp_input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Tensor of size batch x ...
        dim: dimension to apply logsoftmax

    Returns:
    -------
        Tensor of size batch x ...

    """
    max_val = max(input, dim=dim)
    shifted_logits = input - max_val
    logsumexp = max_val + shifted_logits.exp().sum(dim).log()
    return input - logsumexp
