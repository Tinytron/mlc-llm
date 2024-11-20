from typing import List, Optional, Union

from tvm import arith, ir, tir
from tvm.target import Target

from tvm.dlight.base import (
    BlockInfo,
    collect_block_iter_vars_used_in_access_region,
    collect_vars_used_in_prim_expr,
    detect_dominant_read,
    is_broadcast_epilogue,
    normalize_prim_func,
    try_inline_contiguous_spatial,
)
from tvm.dlight.gpu.base import GPUScheduleRule
from tvm.dlight.gpu.gemv import is_gemv, normalize
# from tvm.dlight.gpu.utils import auto_vectorize, get_bytes, get_extent

def get_max_factor(n, factors):
    factors = sorted(factors, reverse=True)
    for factor in factors:
        if n % factor == 0:
            return factor
    return 1

class OpenCLGEMV(GPUScheduleRule):
    """A rule for GEMV and DecodeGEMV."""

    def apply(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None
        if len(block_infos) == 1:
            epilogue = None
        elif len(block_infos) == 2:
            epilogue = block_infos[1]
            if not epilogue.is_injective():
                return None
        else:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            # either [B, S, R] = [B, S, R] * [B, R]
            # or [S, R] = [S, R] * [R]
            return None
        
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None:
            return None

        # sch begin
        matmul_block = block_info.block_rv
        epilogue_block = epilogue.block_rv if epilogue is not None else None
        
        C_local = sch.cache_write(matmul_block, 0, "local")
        A_local = sch.cache_read(matmul_block, 0, "local")
        B_local = sch.cache_read(matmul_block, 1, "local")

        
        n, k = sch.get_loops(matmul_block)
        int_n = int(sch.get(n).extent)
        int_k = int(sch.get(k).extent)

        micro_size_x = 4
        block_size_x = 128
        micro_size_k = 8

        # if int_n >= 4096:
        #     micro_size_k = 4
        
        VecSize = min(get_max_factor(int_n, [1, 2, 4]), micro_size_x)
        Threads_X = min(get_max_factor(int_n // VecSize, [8, 16, 32, 64, 128]), block_size_x)
        VecSize_K = min(get_max_factor(int_k, [1, 2, 4, 8, 16]), micro_size_k)
        # Threads_K = min(get_max_factor(sch.get(k).extent // VecSize_K, [8, 16, 32, 64, 128]), 64)
        
        bx, tx, vx = sch.split(n, [None, Threads_X, VecSize])
        k0, k1 = sch.split(k, [None, VecSize_K])
        sch.reorder(bx, tx, k0, k1, vx)
        sch.set_scope(matmul_block, 0, "local")

        sch.vectorize(vx)

        sch.reverse_compute_at(C_local, tx)
        sch.compute_at(A_local, k0)
        sch.compute_at(B_local, k0)

        l_axis = sch.get_loops(block=C_local)
        # sch.unroll(l_axis[-2])
        sch.vectorize(l_axis[-1])
        
        l_axis = sch.get_loops(block=A_local)
        # sch.unroll(l_axis[-2])
        sch.vectorize(l_axis[-1])
        
        l_axis = sch.get_loops(block=B_local)
        sch.unroll(l_axis[-2])
        sch.vectorize(l_axis[-1])

        sch.unroll(k1)

        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        if epilogue_block is not None:
            sch.reverse_compute_inline(epilogue_block)
            # sch.unroll(o_ur)

        sch.decompose_reduction(matmul_block, k0)

        return sch
