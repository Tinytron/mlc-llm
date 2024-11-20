# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from tvm import tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV
from tvm.script import tir as T

from tvm.dlight.base import analysis, BlockInfo, IterInfo, normalize_prim_func
from tvm.dlight.gpu.base import GPUScheduleRule
from tvm.dlight.gpu.matmul import *

@dataclass
class MatmulConfig:
    block_size_x: int = 8
    block_size_y: int = 8
    micro_size_x: int = 4
    micro_size_y: int = 4
    micro_size_k: int = 8
    unroll: int = 256  # 0 means no unroll

# best gemm
def get_gemm_a8_config():
    """Get the schedule config for the target"""
    return MatmulConfig(
        block_size_x=64,
        block_size_y=1,
        micro_size_x=4,
        micro_size_y=8,
        micro_size_k=4,
        unroll=8,
    )

def get_low_batch_gemv_config():
    """Get the schedule config for the target"""
    return MatmulConfig(
            block_size_x=128,
            block_size_y=1,
            micro_size_x=4,
            micro_size_y=1,
            micro_size_k=8,
            unroll=1,
        )

def get_gemm_a4_config():
    """Get the schedule config for the target"""
    return Matmul.Config(
        block_size_x=128,
        block_size_y=1,
        micro_size_x=4,
        micro_size_y=4,
        micro_size_k=4,
        unroll=4,
    )

class OpenCLMatmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""
    def __init__(self, config=None):
        if config is None:
            config = get_gemm_a4_config()
        self.config = config

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        config = self.config
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)

        main_block_info = get_block_info(sch, main_block)
        iter_infos = main_block_info.iters
        if not get_index_map(block_stmt):
            return None

        # Checks if it's a inner reduction by getting the last matrix's inner Index
        def is_inner_reduction(block_stmt, iter_infos):
            end_it = block_stmt.reads[-1].region[-1].min
            # print(block_stmt.reads[-1].region[-1].min, block_stmt.reads[-1].region[-1], block_stmt.reads[-1], {it.var: it.kind for it in iter_infos})
            return {it.var: it.kind for it in iter_infos}.get(end_it, "O") == "R"

#        if not is_inner_reduction(block_stmt, iter_infos):
        ret = self.sch_outer_reduction(sch, config, main_block, blocks)
        if ret is not None:
            return ret

        raise Exception("Unknown matmul")


    def sch_outer_reduction(
        self,
        sch: tir.Schedule,
        config: MatmulConfig,
        reduction_block: tir.schedule.BlockRV,
        blocks: List[tir.schedule.BlockRV],
    ) -> Optional[tir.Schedule]:

        """Get vectorization factor"""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        # !!!!! very important to remove global iter_var
        normalize_prim_func(sch)
        
        reduction_loops = sch.get_loops(reduction_block)
        # fuse batch
        if len(reduction_loops) == 4:
            # print("reduction_loops")
            mb, ms, n, k = reduction_loops
            m = sch.fuse(mb, ms)
        else:
            m, n, k = reduction_loops

        # patch
        if config.unroll == 8 and sch.get(n).extent < 10240:
            config = get_gemm_a4_config()

        Threads_X, Threads_Y, VecSize, Unroll_M = (
            config.block_size_x,
            config.block_size_y,
            config.micro_size_x,
            config.unroll,
        )
        
        VecSize = min(get_max_factor(sch.get(n).extent, [1, 2, 4]), config.micro_size_x)
        Threads_X = min(get_max_factor(sch.get(n).extent // VecSize, [8, 16, 32, 64, 128]), config.block_size_x)
        VecSize_K = min(get_max_factor(sch.get(k).extent, [1, 2, 4, 8, 16]), config.micro_size_k)
        Threads_K = min(get_max_factor(sch.get(k).extent // VecSize_K, [8, 16, 32, 64, 128]), 64)
        
        matmul_block = reduction_block
        epilogue_block = None
        if blocks[-1] is not matmul_block:
            epilogue_block = blocks[-1]
        for blk in blocks[:-1]:
            if blk is not matmul_block:
                sch.compute_inline(blk)

        # block = sch.reindex(reduction_block, ("read", 0))
        
        if len(reduction_loops) == 4:
            sch.pad_einsum(reduction_block, [1, Unroll_M, 1, 1])
        else:
            sch.pad_einsum(reduction_block, [Unroll_M, 1, 1])

        trans_block = None
        matmul_reindex = None
        if sch.get_producers(matmul_block):
            trans_block = sch.get_producers(matmul_block)[0]
            matmul_reindex = sch.get_consumers(matmul_block)[0]

            # transpose block schedules
            # sch.set_scope(trans_block, 0, "global.texture-1d")
            loops = sch.get_loops(trans_block)
            if len(loops) == 3:
                mb, ms, k = sch.get_loops(trans_block)
                m = sch.fuse(mb, ms)
            else:
                m, k = sch.get_loops(trans_block)
            bk, tk, vk = sch.split(k, [None, Threads_K, VecSize_K])
            bm, tm = sch.split(m, [None, Unroll_M])
            sch.bind(bm, "blockIdx.y")
            sch.bind(bk, "blockIdx.x")
            sch.bind(tm, "threadIdx.y")
            sch.bind(tk, "threadIdx.x")
            sch.reorder(bm, bk, tm, tk, vk)
            sch.vectorize(vk)
        else:
            matmul_reindex = sch.cache_write(matmul_block, 0, "local")

        if epilogue_block is not None:
            sch.compute_inline(matmul_reindex)
            matmul_reindex = epilogue_block

        # matmul block schedule
        # C_local = sch.cache_write(matmul_block, 0, "local")
        A_local = sch.cache_read(matmul_block, 0, "local")
        B_local = sch.cache_read(matmul_block, 1, "local")

        m, n, k = sch.get_loops(matmul_block)
        bx, tx, vx = sch.split(n, [None, Threads_X, VecSize])
        by, ty, vy = sch.split(m, [None, config.block_size_y, config.micro_size_y])
        k0, k1 = sch.split(k, [None, config.micro_size_k])
        sch.reorder(by, bx, ty, tx, k0, vy, k1, vx)
        sch.set_scope(matmul_block, 0, "local")

        sch.vectorize(vx)

        # sch.reverse_compute_at(C_local, tx)
        sch.compute_at(A_local, vy)
        sch.compute_at(B_local, k0)

        # l_axis = sch.get_loops(block=C_local)
        # sch.unroll(l_axis[-2])
        # sch.vectorize(l_axis[-1])
        
        l_axis = sch.get_loops(block=A_local)
        sch.unroll(l_axis[-2])
        sch.vectorize(l_axis[-1])
        
        l_axis = sch.get_loops(block=B_local)
        sch.unroll(l_axis[-2])
        sch.vectorize(l_axis[-1])

        sch.unroll(vy)
        sch.unroll(k1)

        sch.bind(by, "blockIdx.y")
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        if matmul_reindex is not None:
            sch.reverse_compute_at(matmul_reindex, tx)
            o_ur, o_vec = sch.get_loops(matmul_reindex)[-2:]
            sch.vectorize(o_vec)
            # sch.unroll(o_ur)

        sch.decompose_reduction(matmul_block, k0)

        return sch
