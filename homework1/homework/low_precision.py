'''
from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    """
    assert x_quant_4.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        # Let's store all the required information to load the weights from a checkpoint
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # self.register_buffer is used to store the weights in the model, but not as parameters
        # This makes sure weights are put on the correct device when calling `model.to(device)`.
        # persistent=False makes sure the buffer is not saved or loaded. The bignet has a parameters
        # called "weight" that we need to quantize when the model is loaded.
        self.register_buffer(
            "weight_q4",
            torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        # Register a hook to load the weights from a checkpoint. This function reaches deep into
        # PyTorch internals. It makes sure that Linear4Bit._load_state_dict_pre_hook is called
        # every time the model is loaded from a checkpoint. We will quantize the weights in that function.
        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            #weight = state_dict[f"{prefix}weight"]  # noqa: F841
            #del state_dict[f"{prefix}weight"]
            # TODO: Quantize the weights and store them in self.weight_q4 and self.weight_norm
            # raise NotImplementedError()

            key = f"{prefix}weight"
            if key in state_dict:
                # original float32 weights from checkpoint: shape [out, in]
                weight = state_dict[key]
                del state_dict[key]

                # Flatten to 1D for the provided quantizer
                w1d = weight.contiguous().view(-1).to(torch.float32)

                # Quantize
                q4, norm = block_quantize_4bit(w1d, group_size=self._group_size)

                # Store into buffers (correct dtype already)
                self.weight_q4.copy_(q4)
                self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # TODO: Dequantize and call the layer
            # Hint: You can use torch.nn.functional.linear
            #raise NotImplementedError()

            # Dequantize to float32 and reshape back to [out, in]
            w = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(*self._shape)

            # Compute in float32
            y = torch.nn.functional.linear(x.to(torch.float32), w, self.bias)

            # Return in original dtype
            return y.to(x.dtype)


class BigNet4Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the Linear4Bit module for this.
    It is fine to keep all computation in float32.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            #raise NotImplementedError()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        #raise NotImplementedError()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
'''

from pathlib import Path
import torch

from .bignet import BIGNET_DIM, LayerNorm


def pack_3bit(v: torch.Tensor) -> torch.Tensor:
    """
    Pack uint8 values in [0,7] into a uint8 bytearray using 3 bits/value.
    v must be 1D on CPU or GPU.
    """
    assert v.dtype == torch.uint8
    n = v.numel()
    out_nbytes = (n * 3 + 7) // 8
    out = torch.zeros(out_nbytes, dtype=torch.uint8, device=v.device)

    bitpos = torch.arange(n, device=v.device, dtype=torch.int64) * 3
    byte = bitpos // 8
    shift = bitpos % 8

    # We may write across 2 bytes when shift > 5
    out.scatter_add_(0, byte, (v << shift).to(torch.uint8))
    spill = shift > 5
    if spill.any():
        out.scatter_add_(0, byte[spill] + 1, (v[spill] >> (8 - shift[spill])).to(torch.uint8))
    return out


def unpack_3bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    """
    Unpack uint8 packed array into uint8 values in [0,7]
    """
    assert packed.dtype == torch.uint8
    bitpos = torch.arange(n, device=packed.device, dtype=torch.int64) * 3
    byte = bitpos // 8
    shift = bitpos % 8

    a = (packed[byte] >> shift).to(torch.uint16)
    b = torch.zeros_like(a)
    spill = shift > 5
    if spill.any():
        b[spill] = (packed[byte[spill] + 1].to(torch.uint16) << (8 - shift[spill]))
    v = (a | b) & 0x7
    return v.to(torch.uint8)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self._shape = (out_features, in_features)

        # Packed weights: 3 bits per value
        n = out_features * in_features
        nbytes = (n * 3 + 7) // 8
        self.register_buffer("weight_packed", torch.zeros(nbytes, dtype=torch.uint8), persistent=False)

        # Per-row scale (fp16): out_features values
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float16), persistent=False)

        # Optional bias: store fp16 to keep memory tiny
        self.register_buffer("bias_fp16", torch.zeros(out_features, dtype=torch.float16), persistent=False)
        self._has_bias = bias

        # hook to intercept float32 weight from checkpoint
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        w_key = f"{prefix}weight"
        if w_key in state_dict:
            W = state_dict[w_key].to(torch.float32)  # [out, in]
            del state_dict[w_key]

            # quantize per row to 3-bit signed levels in [-3..3]
            out, inn = W.shape
            # scale = maxabs/3 (avoid div0)
            maxabs = W.abs().amax(dim=1).clamp_min(1e-8)
            scale = (maxabs / 3.0).to(torch.float16)
            self.scale.copy_(scale)

            Wq = torch.round((W / maxabs[:, None]) * 3.0).clamp(-3, 3).to(torch.int16)
            # store as unsigned 0..7
            Wu = (Wq + 3).to(torch.uint8).contiguous().view(-1)

            self.weight_packed.copy_(pack_3bit(Wu))

        b_key = f"{prefix}bias"
        if b_key in state_dict:
            b = state_dict[b_key].to(torch.float16)
            del state_dict[b_key]
            if self._has_bias:
                self.bias_fp16.copy_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # unpack + dequantize to float16/float32
        out, inn = self._shape
        n = out * inn
        Wu = unpack_3bit(self.weight_packed, n).view(out, inn).to(torch.int16) - 3  # [-3..3]
        W = (Wu.to(torch.float32) * (self.scale.to(torch.float32)[:, None] / 3.0))

        b = self.bias_fp16.to(torch.float32) if self._has_bias else None
        y = torch.nn.functional.linear(x.to(torch.float32), W, b)
        return y.to(x.dtype)


class LowerBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels, bias=True),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, bias=True),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels, bias=True),
            )

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # dummy param so backward() always has a grad path
        self._dummy = torch.nn.Parameter(torch.zeros(()))

        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y + 0.0 * self._dummy


def load(path: Path | None):
    net = LowerBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
