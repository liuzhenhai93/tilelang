"""MLU590 lowering pipeline: the pass order.

The python passes live in ``tilelang/mlu/transform`` (mirroring
``tilelang/ascend``); the C++ passes live in ``src/mlu``. The pass order
follows the CPU pipeline and omits the CPU-only launch and atomic passes.
``T.Kernel`` becomes a program-index launch: a cluster kernel's
``blockIdx.x`` is ``taskIdX / 4``; a scalar Block kernel's ``blockIdx.x`` is
``taskIdX``. There is no ``threadIdx``.

Pipelining is the manual form (``examples/mlu/flash_attention.py``): an
explicit ring buffer, early ``T.copy`` into the next slot (lowered to
``__memcpy_async``), and one ``sync("io")`` per iteration from
``tilelang.mlu.language``. ``T.Pipelined`` is not used on this backend.
"""

from __future__ import annotations

from tvm import IRModule, s_tir, tirx
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    should_enable_race_check,
    should_force_let_inline,
)

from . import transform as mlu_transform



def MLUPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    """Lower one MLU module from frontend IR through the packed host API."""
    mod = tirx.transform.BindTarget(target)(mod)
    # One-tile-per-IPU kernels (every gemm tile M <= 64) run one task per
    # IPU with whole-fragment ownership, so the launch must not create a
    # four-lane thread axis: the T.Parallel partitioning and the fragment
    # layouts must see a single worker.
    max_gemm_m = 0

    def _scan(fn_body) -> None:
        def visit(node) -> None:
            nonlocal max_gemm_m
            if isinstance(node, tirx.Call):
                callee = getattr(node, "op", None)
                name = str(getattr(callee, "name", "")) if callee is not None else ""
                if name == "tl.mlu_nram_gemm" and len(node.args) >= 4:
                    value = getattr(node.args[3], "value", None)
                    if isinstance(value, int) and value > max_gemm_m:
                        max_gemm_m = value

        post_order_visit(fn_body, visit)

    for fn in mod.functions.values():
        _scan(fn.body)
    mod = tilelang.transform.MaterializeKernelLaunch(
        lower_grid_binding=True,
        lower_thread_binding=False,
        default_threads=1 if 0 < max_gemm_m <= 64 else None,
        unsupported_annotations=["tl.launch_threads", "cluster_dims"],
    )(mod)
    pass_ctx = tilelang.transform.get_pass_context()

    if should_force_let_inline():
        mod = tilelang.transform.LetInline()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.CanonicalizeLegacyReducer()(mod)
    mod = tilelang.transform.VerifyReducerEpoch()(mod)
    # After reducers are canonicalized, and before LowerTileOp, while bodies
    # are still in source order and tile ops still declare access regions.
    mod = tilelang.transform.VerifyBufferInit()(mod)

    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # Before layout inference, so inferred layouts see the pipelined body.
    # Producer stages are global-to-shared copies. Fragment copies are not.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    mod = tilelang.transform.LayoutInference()(mod)
    mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
    LayoutVisual(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = mlu_transform.LowerMLUElementwise()(mod)
    mod = tilelang.transform.VerifyReducerConsumed()(mod)

    mod = tilelang.transform.DecoupleTypeCast()(mod)
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    mod = tilelang.transform.LowerAccessPtr()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.HoistNonRestrictParams()(mod)

    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # BANG has bfloat16_t and __bang_matmul(bfloat16, bfloat16) -> float.
    # Promoting those operands to float would feed the float GEMM mis-typed pointers.
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=False)(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    mod = s_tir.transform.InferFragment()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)
    # LowerClusterStaging off: core-cooperative SRAM form (no MPU role)

    mod = tilelang.transform.MergeIfStmt()(mod)
    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)
    return mod


mlu_pipeline = PassPipeline("mlu", MLUPassPipelineBody)
