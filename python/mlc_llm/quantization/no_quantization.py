"""The no quantization config"""

import functools
from dataclasses import dataclass

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union
from tvm import DataType, DataTypeCode, IRModule, nd, relax, te, tir, topi

from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from mlc_llm.loader import QuantizeMapping

from .utils import (
    compile_quantize_func,
    is_moe_gate,
)

from mlc_llm.support import logging
logger = logging.getLogger(__name__)


@dataclass
class NoQuantize:  # pylint: disable=too-many-instance-attributes
    """Configuration for no quantization but transpose"""

    name: str
    kind: str
    model_dtype: str  # "float16", "float32"

    def __post_init__(self):
        assert self.kind == "no-quant"
        self._func_cache = {}


    def quantize_model(
            self,
            model: nn.Module,
            quant_map: QuantizeMapping,
            name_prefix: str,
        ) -> nn.Module:
        # return model

        class _Mutator(nn.Mutator):
            def __init__(self, config: NoQuantize, quant_map: QuantizeMapping) -> None:
                super().__init__()
                self.config = config
                self.quant_map = quant_map

            def visit_module(self, name: str, node: nn.Module) -> Any:
                if (
                    isinstance(node, nn.Linear)
                    and not is_moe_gate(name, node)
                ):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [weight_name]
                    self.quant_map.map_func[weight_name] = self.config.transpose_weight
                    return NoTransposeLinear.from_linear(node, self.config)
                elif isinstance(node, nn.Embedding):
                    weight_name = f"{name}.weight"
                    self.quant_map.param_map[weight_name] = [weight_name]
                    self.quant_map.map_func[weight_name] = self.config.transpose_weight
                    return TransposeEmbedding.from_embedding(node, self.config)
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def transpose_weight(self, weight: NDArray) -> List[NDArray]:
        device = weight.device

        def _create_func() -> IRModule:
            class Transposer(nn.Module):
                def main(self, weight: nn.Tensor):
                    return [nn.op.permute_dims(weight)]

            class Transposer128(nn.Module):
                def main(self, weight: nn.Tensor):
                    def pack128(weight: te.Tensor) -> te.Tensor:
                        n, k = weight.shape
                        out_shape = (n // 128, k, 128)
                        return te.compute(
                                shape=out_shape,
                                fcompute=lambda ng, k, nc: weight[ng * 128 + nc, k]
                        )
                    
                    return [nn.op.tensor_expr_op(pack128, "pack128", args=[weight])]

            n, k = weight.shape
            if n % 128 == 0:
                mod = Transposer128()
            else:
                mod = Transposer()
            mod, _ = mod.export_tvm(
                spec={"main": {"weight": nn.spec.Tensor(weight.shape, weight.dtype)}}
            )
            return mod

        key = f"({weight.shape}, {weight.dtype}"
        func = self._func_cache.get(key, None)
        if func is None:
            logger.info("Compiling transpose function for key: %s", key)
            func = compile_quantize_func(_create_func(), device)
            self._func_cache[key] = func
        return func(weight)


def te_matmul128(a: te.Tensor, b: te.Tensor) -> te.Tensor:
    output_shape = a.shape[:-1] + [b.shape[0] * 128]
    def matmul_compute(*idx):
        k = te.reduce_axis((0, a.shape[-1]), name="k")
        a_idx = list(idx[:-1]) + [k]
        b_idx = [idx[-1] // 128, k, idx[-1] % 128]
        return te.sum(a(*a_idx) * b(*b_idx), axis=k)

    return te.compute(
        output_shape,
        lambda *idx: matmul_compute(*idx),  # pylint: disable=unnecessary-lambda
        name="matmul",
    )


def te_take(a: te.Tensor, b: te.Tensor) -> te.Tensor:
    output_shape = list(b.shape) + [a.shape[1]]
    def take_compute(*idx):
        b_idx = idx[:-1]
        return a[b(*b_idx) // 128, idx[-1], b(*b_idx) % 128]

    return te.compute(
        output_shape,
        lambda *idx: take_compute(*idx),  # pylint: disable=unnecessary-lambda
        name="take",
    )


class NoTransposeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module without transpose"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        config: NoQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        if out_features % 128 == 0:
            self.weight = nn.Parameter(
                    (out_features // 128, in_features, 128), config.model_dtype if out_dtype is None else out_dtype
                )
        else:
            self.weight = nn.Parameter(
                    (in_features, out_features), config.model_dtype if out_dtype is None else out_dtype
                )
        if bias:
            self.bias = nn.Parameter(
                (out_features,), config.model_dtype if out_dtype is None else out_dtype
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear, config: NoQuantize) -> "NoTransposeLinear":
        return NoTransposeLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
            config=config,
            bias=getattr(linear, "bias", None) is not None,
            out_dtype=linear.out_dtype,
        )

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        if self.out_features % 128 == 0:
            x = nn.op.tensor_expr_op(te_matmul128, "matmul", args=[x, self.weight])
        else:
            x = nn.op.matmul(x, self.weight, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x


class TransposeEmbedding(nn.Module):
    """An nn.Embedding module with transpose"""

    def __init__(self, num: Union[int, tir.Var], dim: int, config: NoQuantize):
        self.num = num
        self.dim = dim
        self.config = config
        if num % 128 == 0:
            self.weight = nn.Parameter(
                (num // 128, dim, 128), config.model_dtype
            )
        else:
            self.weight = nn.Parameter(
                (dim, num), config.model_dtype
            )

    @staticmethod
    def from_embedding(
        embedding: nn.Embedding, config: NoQuantize
    ) -> "TransposeEmbedding":
        num, dim = embedding.weight.shape
        return TransposeEmbedding(num, dim, config)

    def forward(self, x: nn.Tensor):  # pylint: disable=invalid-name
        take = None
        if self.num % 128 == 0:
            take = lambda w, x: nn.op.tensor_expr_op(te_take, "take", args=[w, x])
        else:
            take = lambda w, x: nn.op.permute_dims(nn.op.take(w, x, axis=1))
        
        if x.ndim == 1:
            return take(self.weight, x)
        return nn.op.reshape(
            take(self.weight, nn.op.reshape(x, shape=[-1])),
            shape=[*x.shape, self.dim],
        )

    def lm_head_forward(self, x: nn.Tensor):
        # return nn.op.matmul(x, self.weight, out_dtype="float32")
        if self.out_features % 128 == 0:
            return nn.op.astype(nn.op.tensor_expr_op(te_matmul128, "matmul", args=[x, self.weight]), "float32")
        else:
            return nn.op.matmul(x, self.weight, out_dtype="float32")
