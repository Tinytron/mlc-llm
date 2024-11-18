"""The no quantization config"""

from dataclasses import dataclass

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Type, Union

from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from mlc_llm.loader import QuantizeMapping
from mlc_llm.support import logging

from .utils import (
    compile_quantize_func,
    is_moe_gate,
)

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
                return self.visit(name, node)

        model.to(dtype=self.model_dtype)
        mutator = _Mutator(self, quant_map)
        model = mutator.visit(name_prefix, model)
        return model

    def transpose_weight(self, weight: NDArray) -> List[NDArray]:
        device = weight.device

        def _create_func():
            class Transposer(nn.Module):
                def main(self, weight: nn.Tensor):
                    return [nn.op.permute_dims(weight)]

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
        x = nn.op.matmul(x, self.weight, out_dtype=self.out_dtype)
        if self.bias is not None:
            x = x + self.bias
        return x
