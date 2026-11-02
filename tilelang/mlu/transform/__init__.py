"""MLU-only TIR transform passes (the python-implemented half).

``tilelang/mlu/pipeline.py`` owns the pass ORDER; the python passes live
here, mirroring ``tilelang/ascend/transform``. The C++ passes live in
``src/mlu``.
"""

from __future__ import annotations

import os
import sys

from tvm import IRModule, s_tir, tirx
from tvm.ir import structural_equal
from tvm.target import Target
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

import tilelang
from tilelang.backend.pass_pipeline import PassPipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    should_enable_race_check,
    should_force_let_inline,
)


def _const_int(expr) -> int | None:
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    return None


def _collect_loop_vars(fn_body) -> list:
    """Every For loop variable in the function, for loop-variant checks."""
    vars_found: list = []

    def visit(node) -> None:
        if isinstance(node, tirx.For):
            vars_found.append(node.loop_var)

    post_order_visit(fn_body, visit)
    return vars_found


def _uses_sync_io(fn_body) -> bool:
    """True when the kernel manages GDRAM load waits with __sync_io itself."""
    found = [False]

    def visit(node) -> None:
        if isinstance(node, tirx.Call):
            name = getattr(node, "name", None)
            if name is None and len(node.args) >= 1:
                a0 = node.args[0]
                if isinstance(a0, tirx.StringImm):
                    name = a0.value
            if name == "__sync_io":
                found[0] = True

    post_order_visit(fn_body, visit)
    return found[0]


def LowerMLUElementwise():
    """Replace contiguous NRAM scalar loops with BANG NFU calls.

    Unit-stride panels that only move bytes become one ``__memcpy``. A fused
    store of ``cast(output / row_sum)`` divides each row in place, casts the
    owned panel once, then copies that panel in one transfer. Other shapes
    keep the scalar loop.
    """

    def pass_fn(func: tirx.PrimFunc, mod, ctx):
        del mod, ctx
        all_loop_vars = _collect_loop_vars(func.body)
        # The gemm A/C fragments: loads into these are issued async (the old
        # split-pass behavior). NOTE (2026-11-02): __memcpy_async LOADS have
        # NO fence on this stack -- sync("io"/"move"/"cluster") and even the
        # raw pio-tagged barrier all return without waiting (probe_fence.py
        # / probe_syncload.py / probe_barpio.py). An async load is only
        # drained by a LATER SYNC __memcpy. Do not extend this set: an
        # unfenced async load races its consumer (stale filters -> wrong
        # output).
        gemm_ac_vars: set = set()

        def scan_gemm(node) -> None:
            if isinstance(node, tirx.Call) and len(node.args) >= 4:
                callee = getattr(node, "op", None)
                name = str(getattr(callee, "name", "")) if callee is not None else ""
                if name == "tl.mlu_nram_gemm":
                    for slot in (1, 2):
                        acc = node.args[slot]
                        if isinstance(acc, tirx.Call) and len(acc.args) >= 2:
                            gemm_ac_vars.add(acc.args[1])

        post_order_visit(func.body, scan_gemm)
        loop_var_stack: list[tirx.Var] = []

        def is_private(buffer) -> bool:
            return buffer.scope() in ("local.fragment", "local", "local.var")

        def is_shared(buffer) -> bool:
            return buffer.scope() in ("shared", "shared.dyn")

        def uses_var(expr, var: tirx.Var) -> bool:
            found = False

            def visit(node) -> None:
                nonlocal found
                if isinstance(node, tirx.Var) and node.same_as(var):
                    found = True

            post_order_visit(expr, visit)
            return found

        def with_body(loop, body):
            return tirx.For(
                loop.loop_var,
                loop.min,
                loop.extent,
                loop.kind,
                body,
                loop.thread_binding,
                loop.annotations,
                loop.step,
            )

        def flat_index(buffer, indices):
            indices = list(indices)
            shape = list(buffer.shape)
            if len(indices) == 1:
                return indices[0]
            if len(indices) != len(shape):
                return None
            strides = list(buffer.strides) if buffer.strides is not None else []
            if len(strides) == len(shape):
                offset = indices[0] * strides[0]
                for index, stride in zip(indices[1:], strides[1:]):
                    offset = offset + index * stride
                return offset
            if len(strides) != 0:
                return None
            offset = indices[0]
            for dim, index in zip(shape[1:], indices[1:]):
                offset = offset * dim + index
            return offset

        def element_ptr(buffer, offset, length, rw_mask):
            return tirx.op.tvm_access_ptr(
                tirx.op.type_annotation(buffer.dtype),
                buffer.data,
                offset,
                tirx.IntImm("int32", int(length)),
                tirx.IntImm("int32", rw_mask),
            )

        direction_names = {0: "GDRAM2NRAM", 1: "NRAM2GDRAM"}

        def memcpy_stmt(dst, dst_offset, src, src_offset, nbytes, direction, async_copy=False):
            dst_count = nbytes // (dst.dtype.bits // 8)
            src_count = nbytes // (src.dtype.bits // 8)
            direction_name = (
                direction if isinstance(direction, str) else direction_names[direction]
            )
            name = "__memcpy_async" if async_copy else "__memcpy"
            return tirx.Evaluate(
                tirx.call_extern(
                    "void",
                    name,
                    element_ptr(dst, dst_offset, dst_count, 2),
                    element_ptr(src, src_offset, src_count, 1),
                    tirx.IntImm("int32", nbytes),
                    tirx.call_extern("int32", direction_name),
                )
            )

        def prove_same(lhs, rhs) -> bool:
            from tvm.arith import Analyzer
            from tvm.ir import structural_equal

            analyzer = Analyzer()
            if analyzer.can_prove_equal(lhs, rhs):
                return True
            return structural_equal(analyzer.simplify(lhs), analyzer.simplify(rhs))

        def unit_step(loop) -> bool:
            return loop.step is None or _const_int(loop.step) == 1

        def sync_stmt(name):
            return tirx.Evaluate(tirx.call_extern("void", name))


        def try_memcpy(stmt):
            import os as _os
            _dbg = _os.environ.get("TL_MLU_DEBUG_MATCH") == "1"
            inner = stmt.body if isinstance(stmt.body, tirx.For) else None
            if inner is None:
                # Single-loop form (e.g. a one-row tile copy whose only
                # extent-1 dimension was dropped by the tile-op expansion):
                # rows == 1 and the single loop is the contiguous dimension.
                if not isinstance(stmt.body, tirx.BufferStore):
                    if _dbg:
                        print(f"[cpy] skip: body={type(stmt.body).__name__}", file=sys.stderr)
                    return None
                store = stmt.body
            else:
                if not isinstance(inner.body, tirx.BufferStore):
                    if _dbg:
                        print(f"[cpy] skip: body={type(inner.body).__name__}", file=sys.stderr)
                    return None
                store = inner.body
            if store.predicate is not None or not isinstance(store.value, tirx.BufferLoad):
                if _dbg:
                    print(f"[cpy] skip: store={type(store.value).__name__}", file=sys.stderr)
                return None
            load = store.value
            if getattr(load, "predicate", None) is not None:
                return None
            if str(store.buffer.dtype) != str(load.buffer.dtype):
                if _dbg:
                    print(f"[cpy] skip: dtype {store.buffer.dtype} vs {load.buffer.dtype}", file=sys.stderr)
                return None
            if _dbg:
                print(f"[cpy] reach: store={store.buffer.name}({store.buffer.scope()}) "
                      f"load={load.buffer.name}({load.buffer.scope()})", file=sys.stderr)
            if is_private(store.buffer) and load.buffer.scope() == "global":
                direction = 0
                private, global_buf = store.buffer, load.buffer
                private_index = flat_index(private, store.indices)
                global_indices = list(load.indices)
            elif store.buffer.scope() == "global" and is_private(load.buffer):
                direction = 1
                private, global_buf = load.buffer, store.buffer
                private_index = flat_index(private, load.indices)
                global_indices = list(store.indices)
            else:
                return None
            if private_index is None:
                return None
            rows = _const_int(stmt.extent) if inner is not None else 1
            cols = (
                _const_int(inner.extent) if inner is not None else _const_int(stmt.extent)
            )
            if (
                rows is None
                or cols is None
                or rows <= 0
                or cols <= 0
                or inner is not None
                and (_const_int(inner.min) != 0 or not unit_step(inner))
                or not unit_step(stmt)
            ):
                return None
            nbytes_head = rows * cols * (store.buffer.dtype.bits // 8)
            if is_shared(store.buffer) and load.buffer.scope() == "global":
                # cluster-SRAM producer fill: GDRAM -> SRAM, async (the
                # producer's own sync("io") waits for it)
                return memcpy_stmt(
                    store.buffer, flat_index(store.buffer, store.indices),
                    load.buffer, flat_index(load.buffer, load.indices),
                    nbytes_head, "GDRAM2SRAM", True,
                )
            if is_private(store.buffer) and is_shared(load.buffer):
                # cluster-SRAM consumption: SRAM -> NRAM fragment, async
                return memcpy_stmt(
                    store.buffer, flat_index(store.buffer, store.indices),
                    load.buffer, flat_index(load.buffer, load.indices),
                    nbytes_head, "SRAM2NRAM", True,
                )
            from tvm.tirx.stmt_functor import substitute

            glinear = flat_index(global_buf, global_indices)
            if glinear is None:
                return None
            if inner is None:
                # Single-loop form: the one loop walks the contiguous dim.
                element = stmt.loop_var
                vmap = {stmt.loop_var: stmt.min}
            else:
                element = stmt.loop_var * tirx.IntImm("int32", cols) + inner.loop_var
                vmap = {stmt.loop_var: stmt.min, inner.loop_var: inner.min}
            element_start = substitute(element, vmap)
            local_start = substitute(private_index, vmap)
            # Software pipelining may add a leading buffer-version dimension to
            # the private tile (``k[v, i, j]``).  Accept it as long as the
            # version offset is invariant across the matched loops, i.e. the
            # private flat index is ``element + c`` for a loop-invariant ``c``.
            if not prove_same(private_index, element + (local_start - element_start)):
                return None
            global_start = substitute(glinear, vmap)
            if not prove_same(glinear, global_start + (element - element_start)):
                return None
            nbytes = rows * cols * (store.buffer.dtype.bits // 8)
            # T.copy(..., sync=False) sets mlu_sync=0 on the outermost loop.
            # That one copy is __memcpy_async; the caller still owes sync("io").
            marker = None
            ann = getattr(stmt, "annotations", None)
            if ann is not None and "mlu_sync" in ann:
                marker = ann["mlu_sync"]
            explicit_async = _const_int(marker) == 0
            # An explicit ring buffer (e.g. ``k[ko % 2]``) keeps
            # ``versions * rows * cols`` elements and each copy targets one
            # slot at a ``rows * cols``-aligned offset. Load copies into such
            # a slot become ``__memcpy_async`` so the DMA overlaps compute;
            # the source's ``T.sync("io")`` waits for them. Stores to global
            # stay synchronous.
            total = 1
            for dim in private.shape:
                const = _const_int(dim)
                if const is None:
                    total = None
                    break
                total *= const
            tile = rows * cols
            versioned = (
                os.environ.get("TL_MLU_SYNC_COPY") != "1"
                and total is not None
                and tile > 0
                and total % tile == 0
                and total // tile > 1
            )
            if not versioned:
                if direction == 0:
                    # Async only for loads into gemm A/C fragments (the old
                    # split-pass behavior): the kernel's own __sync_io waits
                    # for them. Everything else stays synchronous so the
                    # dependent NFU ops remain inline instead of being pushed
                    # onto the async stream, whose context switches
                    # save/restore the whole NRAM image per task.
                    dst_var = private.data
                    load_async = explicit_async or dst_var in gemm_ac_vars
                    return memcpy_stmt(
                        private, local_start, global_buf, global_start, nbytes, 0,
                        load_async,
                    )
                return memcpy_stmt(
                    global_buf, global_start, private, local_start, nbytes, 1,
                    explicit_async,
                )
            if direction == 0:
                return memcpy_stmt(
                    private, local_start, global_buf, global_start, nbytes, 0, True
                )
            return memcpy_stmt(
                global_buf, global_start, private, local_start, nbytes, 1,
                explicit_async,
            )

        def try_transpose(stmt):
            """A stride-1 scan of one private buffer stored transposed into
            another becomes one ``__bang_transpose``.

            The builder flattens ``dst[j, i] = src[i, j]`` into a flat scan
            where one side keeps the linear index and the other side is the
            composite ``FloorMod(f, W) * H + FloorDiv(f, W)``.
            """
            if not isinstance(stmt, tirx.For) or not isinstance(stmt.body, tirx.BufferStore):
                return None
            store = stmt.body
            if store.predicate is not None or not isinstance(store.value, tirx.BufferLoad):
                return None
            load = store.value
            if getattr(load, "predicate", None) is not None:
                return None
            if str(store.buffer.dtype) != str(load.buffer.dtype):
                return None
            if not (is_private(store.buffer) and is_private(load.buffer)):
                return None
            if len(store.indices) != 1 or len(load.indices) != 1:
                return None
            store_total = 1
            for dim in store.buffer.shape:
                value = _const_int(dim)
                if value is None:
                    return None
                store_total *= value
            load_total = 1
            for dim in load.buffer.shape:
                value = _const_int(dim)
                if value is None:
                    return None
                load_total *= value
            flat_side = None
            composite_side = None
            flat_base = None
            composite_base = None

            def strip_invariant_addend(index):
                """Split ``index`` into (core, invariant addend).

                A software-pipelined ring buffer indexes its slot as
                ``version * tile + element``; the version term is invariant
                across the matched loop and becomes the side's base offset.
                Nested Add chains are flattened first so the version term is
                found even when it sits mid-tree (e.g. a chunked transpose
                writes ``(chunk * tile + element) * stride + col``).
                """
                terms = []

                def flatten(node) -> None:
                    if isinstance(node, tirx.Add):
                        flatten(node.a)
                        flatten(node.b)
                    else:
                        terms.append(node)

                flatten(index)
                if len(terms) <= 1:
                    return index, None

                def uses_loop_var(node) -> bool:
                    found = [False]

                    def visit_var(node) -> None:
                        if isinstance(node, tirx.Var) and node.same_as(stmt.loop_var):
                            found[0] = True

                    post_order_visit(node, visit_var)
                    return found[0]

                core_terms = [t for t in terms if uses_loop_var(t)]
                inv_terms = [t for t in terms if not uses_loop_var(t)]
                if not core_terms or not inv_terms:
                    return index, None
                core = core_terms[0]
                for term in core_terms[1:]:
                    core = core + term
                base = inv_terms[0]
                for term in inv_terms[1:]:
                    base = base + term
                return core, base

            for side, index in ((load, load.indices[0]), (store, store.indices[0])):
                core, base = strip_invariant_addend(index)
                if structural_equal(core, stmt.loop_var):
                    if flat_side is not None:
                        return None
                    flat_side = side
                    flat_base = base
                elif isinstance(core, tirx.Add):
                    mul_part, div_part = core.a, core.b
                    if isinstance(div_part, tirx.Mul) and isinstance(mul_part, tirx.FloorDiv):
                        # Add flattening may hand back the terms in either
                        # order; accept both.
                        mul_part, div_part = div_part, mul_part
                    if (
                        isinstance(mul_part, tirx.Mul)
                        and isinstance(mul_part.a, tirx.FloorMod)
                        and isinstance(div_part, tirx.FloorDiv)
                        and structural_equal(mul_part.a.a, stmt.loop_var)
                        and structural_equal(div_part.a, stmt.loop_var)
                    ):
                        width = _const_int(mul_part.a.b)
                        height = _const_int(mul_part.b)
                        if (
                            width is None
                            or height is None
                            or width <= 0
                            or height <= 0
                            or _const_int(div_part.b) != width
                        ):
                            return None
                        if composite_side is not None:
                            return None
                        composite_side = side
                        composite_base = base
                    else:
                        return None
                else:
                    return None
            length = height * width
            # Ring buffers hold several versions of the tile, so each side's
            # total must be a whole multiple of the per-slot H*W.
            if store_total % length != 0 or load_total % length != 0:
                return None
            if flat_side is None or composite_side is None:
                return None
            if _const_int(stmt.min) != 0 or _const_int(stmt.extent) != length:
                return None
            # The instruction transposes src [h, w] into dst [w, h]. The flat
            # side is the [H, W] view; the composite side is [W, H]. A ring
            # slot's invariant base offset rides on each side's pointer.
            if flat_side is load:
                inst_height, inst_width = height, width
            else:
                inst_height, inst_width = width, height
            store_base = composite_base if composite_side is store else flat_base
            load_base = flat_base if flat_side is load else composite_base
            dst_ptr = element_ptr(
                store.buffer,
                store_base if store_base is not None else 0,
                length,
                2,
            )
            src_ptr = element_ptr(
                load.buffer,
                load_base if load_base is not None else 0,
                length,
                1,
            )
            transpose = tirx.Evaluate(
                tirx.call_extern(
                    "void",
                    "__bang_transpose",
                    dst_ptr,
                    src_ptr,
                    tirx.IntImm("int32", int(inst_height)),
                    tirx.IntImm("int32", int(inst_width)),
                )
            )
            # The stream transpose runs on its own (move) queue; consumers on
            # the compute queue must wait for it explicitly. __sync_move
            # drains only the move queue, so in-flight TFU matmuls keep
            # running while the transpose lands; __sync_compute would drain
            # them and serialize the whole softmax chain.
            # TL_MLU_TRANSPOSE_SYNC=off drops the barrier (diagnostics only:
            # same-queue consumers stay ordered, cross-queue ones race).
            if os.environ.get("TL_MLU_TRANSPOSE_SYNC") == "off":
                return transpose
            return tirx.SeqStmt([transpose, sync_stmt("__sync_move")])

        def try_epilogue(stmt):
            inner = stmt.body if isinstance(stmt.body, tirx.For) else None
            if inner is None:
                return None
            leaf = inner.body
            bind = None
            if (
                isinstance(leaf, tirx.SeqStmt)
                and len(leaf.seq) == 2
                and isinstance(leaf.seq[0], tirx.Bind)
                and isinstance(leaf.seq[1], tirx.BufferStore)
            ):
                bind, store = leaf.seq[0], leaf.seq[1]
            elif isinstance(leaf, tirx.BufferStore):
                store = leaf
            else:
                return None
            if store.predicate is not None or store.buffer.scope() != "global":
                return None
            expr = store.value
            if bind is not None:
                from tvm.tirx.stmt_functor import substitute

                expr = substitute(expr, {bind.var: bind.value})
            cast_to = None
            if isinstance(expr, tirx.Cast):
                cast_to = str(expr.dtype)
                expr = expr.value
            if type(expr).__name__ != "Div":
                return None
            numer, denom = expr.a, expr.b
            if (
                not isinstance(numer, tirx.BufferLoad)
                or len(numer.indices) != 1
                or not is_private(numer.buffer)
                or str(numer.buffer.dtype) != "float32"
                or not isinstance(denom, tirx.BufferLoad)
                or len(denom.indices) != 1
                or not is_private(denom.buffer)
            ):
                return None
            rows = _const_int(stmt.extent)
            cols = _const_int(inner.extent)
            if (
                rows is None
                or cols is None
                or rows <= 0
                or cols <= 0
                or _const_int(inner.min) != 0
                or not unit_step(stmt)
                or not unit_step(inner)
            ):
                return None
            nbytes_head = rows * cols * (store.buffer.dtype.bits // 8)
            if is_shared(store.buffer) and load.buffer.scope() == "global":
                # cluster-SRAM producer fill: GDRAM -> SRAM, async (the
                # producer's own sync("io") waits for it)
                return memcpy_stmt(
                    store.buffer, flat_index(store.buffer, store.indices),
                    load.buffer, flat_index(load.buffer, load.indices),
                    nbytes_head, "GDRAM2SRAM", True,
                )
            if is_private(store.buffer) and is_shared(load.buffer):
                # cluster-SRAM consumption: SRAM -> NRAM fragment, async
                return memcpy_stmt(
                    store.buffer, flat_index(store.buffer, store.indices),
                    load.buffer, flat_index(load.buffer, load.indices),
                    nbytes_head, "SRAM2NRAM", True,
                )
            count = rows * cols
            if count % 64 != 0:
                return None
            from tvm.ir import structural_equal
            from tvm.tirx.stmt_functor import substitute

            element = stmt.loop_var * tirx.IntImm("int32", cols) + inner.loop_var
            if not structural_equal(numer.indices[0], element):
                return None
            row_index = denom.indices[0]
            period = None
            if isinstance(row_index, tirx.FloorDiv) and (
                structural_equal(row_index.a, element) or prove_same(row_index.a, element)
            ):
                period = _const_int(row_index.b)
            elif structural_equal(row_index, stmt.loop_var) and cols % 64 == 0:
                period = cols
            if period is None or period % 64 != 0 or count % period != 0:
                return None
            glinear = flat_index(store.buffer, list(store.indices))
            if glinear is None:
                return None
            vmap = {stmt.loop_var: stmt.min, inner.loop_var: inner.min}
            local_start = substitute(element, vmap)
            global_start = substitute(glinear, vmap)
            if not prove_same(glinear, global_start + (element - local_start)):
                return None
            row = tirx.Var("out_row", "int32")
            origin = local_start + row * tirx.IntImm("int32", period)
            reciprocal = tirx.Div(
                tirx.FloatImm("float32", 1.0),
                tirx.BufferLoad(denom.buffer, [tirx.FloorDiv(origin, tirx.IntImm("int32", period))]),
            )
            scale = tirx.Evaluate(
                tirx.call_extern(
                    "void",
                    "__tl_mlu_ew_f32_mul_scalar",
                    element_ptr(numer.buffer, origin, period, 2),
                    element_ptr(numer.buffer, origin, period, 1),
                    reciprocal,
                    tirx.IntImm("int32", period),
                )
            )
            row_loop = tirx.For(
                row,
                tirx.IntImm("int32", 0),
                tirx.IntImm("int32", count // period),
                tirx.ForKind.SERIAL,
                scale,
            )
            if cast_to == "bfloat16":
                tmp = tirx.decl_buffer((count,), "bfloat16", scope="local", name="out_cast")
                cast = tirx.Evaluate(
                    tirx.call_extern(
                        "void",
                        "__bang_float2bfloat16",
                        element_ptr(tmp, 0, count, 2),
                        element_ptr(numer.buffer, local_start, count, 1),
                        tirx.IntImm("int32", count),
                    )
                )
                copy = memcpy_stmt(store.buffer, global_start, tmp, 0, count * 2, 1)
                return tirx.SeqStmt([row_loop, tirx.AllocBuffer(tmp, {}), cast, copy])
            if cast_to is None and str(store.buffer.dtype) == "float32":
                copy = memcpy_stmt(
                    store.buffer, global_start, numer.buffer, local_start, count * 4, 1
                )
                return tirx.SeqStmt([row_loop, copy])
            return None

        def lower(stmt):
            if not isinstance(stmt, tirx.For):
                return None
            copied = try_memcpy(stmt)
            if copied is not None:
                return copied
            epilogue = try_epilogue(stmt)
            if epilogue is not None:
                return epilogue
            transposed = try_transpose(stmt)
            if transposed is not None:
                return transposed
            # A loop that updates several vectors, such as row_max and row_scale,
            # is one SeqStmt. Lower each store on its own and keep the order.
            nested = stmt.body if isinstance(stmt.body, tirx.For) else None
            leaf = nested.body if nested is not None else stmt.body
            if isinstance(leaf, tirx.SeqStmt):
                parts = []
                for item in leaf.seq:
                    if not isinstance(item, tirx.BufferStore):
                        return None
                    body = with_body(nested, item) if nested is not None else item
                    lowered = lower(with_body(stmt, body))
                    if lowered is None:
                        return None
                    parts.append(lowered)
                return parts[0] if len(parts) == 1 else tirx.SeqStmt(parts)
            inner = nested
            store = leaf
            if not isinstance(store, tirx.BufferStore):
                return None
            outer_count = _const_int(stmt.extent)
            inner_count = _const_int(inner.extent) if inner is not None else 1
            if (
                outer_count is None
                or outer_count <= 0
                or inner is not None
                and (_const_int(inner.min) != 0 or inner_count is None or inner_count <= 0)
            ):
                return None
            if len(store.indices) != 1 or not is_private(store.buffer):
                return None

            def base_index(index):
                if isinstance(index, tirx.Ramp):
                    lanes = _const_int(index.lanes)
                    if _const_int(index.stride) != 1 or lanes is None:
                        return None
                    return index.base, lanes
                return index, 1

            store_index = base_index(store.indices[0])
            if store_index is None:
                return None
            store_base, lanes = store_index
            loop_count = outer_count * inner_count
            loop_offset = stmt.loop_var * inner_count
            if inner is not None:
                loop_offset = loop_offset + inner.loop_var
            count = loop_count * lanes
            if count % 64 != 0 or not uses_var(store_base, stmt.loop_var):
                return None
            if inner is not None and not uses_var(store_base, inner.loop_var):
                return None

            from tvm.ir import structural_equal
            from tvm.tirx.stmt_functor import substitute

            def divisible(expr, divisor: int) -> bool:
                if isinstance(expr, tirx.IntImm):
                    return int(expr.value) % divisor == 0
                if isinstance(expr, tirx.Mul):
                    return divisible(expr.a, divisor) or divisible(expr.b, divisor)
                if isinstance(expr, tirx.Add):
                    return divisible(expr.a, divisor) and divisible(expr.b, divisor)
                return False

            def broadcast_period(index) -> int | None:
                if not isinstance(index, tirx.FloorDiv):
                    return None
                period = _const_int(index.b)
                if period is None or period <= 0 or period % 64 != 0:
                    return None
                if not structural_equal(index.a, store_base):
                    return None
                return period

            def find_period(expr) -> int | None:
                found: int | None = None
                rejected = False

                def visit(node) -> None:
                    nonlocal found, rejected
                    if rejected or not isinstance(node, tirx.BufferLoad) or len(node.indices) != 1:
                        return
                    index = node.indices[0]
                    normalized = base_index(index)
                    if normalized is not None and normalized[1] == lanes and structural_equal(normalized[0], store_base):
                        return
                    if not uses_var(index, stmt.loop_var) and (inner is None or not uses_var(index, inner.loop_var)):
                        return
                    period = broadcast_period(index)
                    if period is None or (found is not None and found != period):
                        rejected = True
                        return
                    found = period

                post_order_visit(expr, visit)
                return None if rejected else found

            # A row broadcast such as ``row_max[(i * 64 + j) // 64]`` is constant
            # for one NFU-sized row and changes on the next row. Emit one vector
            # chain per row instead of one scalar for the whole loop.
            def contiguous_start():
                if lanes != 1:
                    return None
                expected = stmt.loop_var if inner is None else (
                    stmt.loop_var * tirx.IntImm("int32", inner_count) + inner.loop_var
                )
                if not structural_equal(store_base, expected):
                    return None
                vmap = {stmt.loop_var: stmt.min}
                if inner is not None:
                    vmap[inner.loop_var] = inner.min
                return substitute(store_base, vmap)

            period = find_period(store.value)
            chunk_origin = None
            emit_count = count
            row_var = None
            expand_rows = None
            row_len = None
            # Row-broadcast lowering: "row" (default) keeps the per-row chain;
            # "expand"/"cycle" opt into whole-panel instructions. __bang_expand_*
            # needs BANG arch >= 613 and is unavailable on mtp_592.
            bcast_op = os.environ.get("TL_MLU_BCAST_OP", "row")
            if os.environ.get("TL_MLU_DEBUG_MATCH"):
                import sys
                print(
                    f"[loop] buf={store.buffer.name} min={stmt.min} ext={_const_int(stmt.extent)} "
                    f"body={type(stmt.body).__name__} "
                    f"inner_ext={_const_int(inner_count) if inner else None} lanes={lanes} "
                    f"period={period} store_base={store_base}",
                    file=sys.stderr,
                )
            if period is None:
                if _const_int(stmt.min) != 0:
                    # The cluster pass has already narrowed this loop. One vector
                    # call covers the owned elements; the pointer starts at min.
                    start = contiguous_start()
                    if start is None or not divisible(start, 64):
                        return None
                    chunk_origin = start
            else:
                if lanes != 1 or count % period != 0:
                    return None
                start = contiguous_start()
                if start is None or not divisible(start, period):
                    return None
                if bcast_op == "expand" and period % 64 == 0:
                    # The FloorDiv broadcast maps 1:1 onto __bang_expand_*:
                    # dst[i] op src1[i / expand_times]. One whole-panel call
                    # replaces the per-row call chain.
                    chunk_origin = start
                    row_len = period
                    expand_rows = count // period
                else:
                    row_var = tirx.Var("row", "int32")
                    chunk_origin = start + row_var * tirx.IntImm("int32", period)
                    emit_count = period

            # A row broadcast of the form ``row_max[i]`` — the outer loop
            # variable itself indexing one scalar per NFU row — would cost one
            # call chain per row. When every loop-dependent operand is either
            # the panel itself or such a row vector, the BANG ``__bang_expand_*``
            # ops cover the whole panel in one instruction.
            def panel_broadcasts_ok(expr) -> bool:
                ok = True

                def visit(node) -> None:
                    nonlocal ok
                    if not ok or not isinstance(node, tirx.BufferLoad) or len(node.indices) != 1:
                        return
                    index = node.indices[0]
                    normalized = base_index(index)
                    if (
                        normalized is not None
                        and normalized[1] == lanes
                        and structural_equal(normalized[0], store_base)
                    ):
                        return
                    if not uses_var(index, stmt.loop_var) and (
                        inner is None or not uses_var(index, inner.loop_var)
                    ):
                        return
                    if not structural_equal(index, stmt.loop_var):
                        ok = False

                post_order_visit(expr, visit)
                return ok

            if period is None and lanes == 1 and inner is not None and bcast_op != "row":
                candidate = _const_int(inner_count)
                start = contiguous_start()
                if (
                    candidate is not None
                    and candidate > 0
                    and count % candidate == 0
                    and start is not None
                    and divisible(start, 64)
                    and panel_broadcasts_ok(store.value)
                ):
                    chunk_origin = start
                    row_len = candidate
                    expand_rows = count // row_len

            # A column-indexed broadcast on a transposed panel flattens to
            # ``row_max[flat % period]``. The BANG cycle ops apply
            # ``seg[i % seg_count]``, which is exactly this form, so the whole
            # panel lowers to one instruction instead of a scalar gather.
            cycle_period = None

            def cycle_period_of(expr):
                found = None
                rejected = False

                def visit(node) -> None:
                    nonlocal found, rejected
                    if rejected or not isinstance(node, tirx.BufferLoad) or len(node.indices) != 1:
                        return
                    index = node.indices[0]
                    normalized = base_index(index)
                    if (
                        normalized is not None
                        and normalized[1] == lanes
                        and structural_equal(normalized[0], store_base)
                    ):
                        return
                    if not uses_var(index, stmt.loop_var) and (
                        inner is None or not uses_var(index, inner.loop_var)
                    ):
                        return
                    if not (
                        isinstance(index, tirx.FloorMod)
                        and structural_equal(index.a, store_base)
                    ):
                        rejected = True
                        return
                    candidate = _const_int(index.b)
                    if (
                        candidate is None
                        or candidate <= 0
                        or candidate % 64 != 0
                        or (found is not None and found != candidate)
                    ):
                        rejected = True
                        return
                    found = candidate

                post_order_visit(expr, visit)
                return None if rejected else found

            if period is None and expand_rows is None and lanes == 1:
                candidate = cycle_period_of(store.value)
                if candidate is not None:
                    start = contiguous_start()
                    if start is not None and divisible(start, candidate):
                        chunk_origin = start
                        cycle_period = candidate

            def access(buffer, index, rw_mask: int):
                normalized = base_index(index)
                if normalized is None:
                    return None
                base, source_lanes = normalized
                if source_lanes != lanes:
                    return None
                if chunk_origin is not None:
                    if not structural_equal(base, store_base):
                        return None
                    offset = chunk_origin
                    length = emit_count
                else:
                    offset = base - loop_offset * lanes
                    # The map must be contiguous: offset has to be invariant
                    # in the loop vars. Prove it arithmetically (a fused
                    # parallel loop flattens to floor-div/floor-mod forms
                    # that are syntactically loop-dependent but constant).
                    loop_vars = [stmt.loop_var] + (
                        [inner.loop_var] if inner is not None else []
                    )
                    from tvm import arith as _arith

                    linear = _arith.detect_linear_equation(offset, loop_vars)
                    _an = _arith.Analyzer()
                    if linear is None or any(
                        not isinstance(_zero := _an.simplify(c), tirx.IntImm)
                        or int(_zero.value) != 0
                        for c in list(linear)[1:]
                    ):
                        return None
                    length = count
                return tirx.op.tvm_access_ptr(
                    tirx.op.type_annotation(buffer.dtype),
                    buffer.data,
                    offset,
                    tirx.IntImm("int32", length),
                    tirx.IntImm("int32", rw_mask),
                )

            dst = access(store.buffer, store.indices[0], 2)
            if dst is None:
                return None
            dtype = store.buffer.dtype
            suffix = "bf16" if str(dtype) == "bfloat16" else "f32"
            calls: list[tirx.Stmt] = []

            def vector(expr):
                if isinstance(expr, tirx.Cast) and isinstance(expr.value, tirx.BufferLoad):
                    expr = expr.value
                if not isinstance(expr, tirx.BufferLoad) or len(expr.indices) != 1:
                    return None
                if expr.buffer.dtype != dtype or not is_private(expr.buffer):
                    return None
                normalized = base_index(expr.indices[0])
                if normalized is None or normalized[1] != lanes:
                    return None
                if not structural_equal(normalized[0], store_base):
                    return None
                return access(expr.buffer, expr.indices[0], 1)

            def vector_float(expr):
                if not isinstance(expr, tirx.BufferLoad) or len(expr.indices) != 1:
                    return None
                if str(expr.buffer.dtype) != "float32" or not is_private(expr.buffer):
                    return None
                normalized = base_index(expr.indices[0])
                if normalized is None or normalized[1] != lanes:
                    return None
                if not structural_equal(normalized[0], store_base):
                    return None
                return access(expr.buffer, expr.indices[0], 1)

            def scalar(expr):
                if isinstance(expr, (tirx.FloatImm, tirx.IntImm)):
                    return tirx.Cast(dtype, expr) if str(dtype) == "bfloat16" else expr
                if isinstance(expr, tirx.BufferLoad) and len(expr.indices) == 1:
                    if expr.buffer.dtype != dtype or not is_private(expr.buffer):
                        return None
                    if (
                        expand_rows is None
                        and period is not None
                        and chunk_origin is not None
                        and broadcast_period(expr.indices[0]) == period
                    ):
                        return tirx.BufferLoad(
                            expr.buffer,
                            [tirx.floordiv(chunk_origin, tirx.IntImm("int32", period))],
                        )
                    if not uses_var(expr.indices[0], stmt.loop_var) and (
                        inner is None or not uses_var(expr.indices[0], inner.loop_var)
                    ):
                        return expr
                return None

            def row_operand(expr):
                """Access pointer for a row-broadcast load (expand operand)."""
                if expand_rows is None:
                    return None
                if not isinstance(expr, tirx.BufferLoad) or len(expr.indices) != 1:
                    return None
                index = expr.indices[0]
                if isinstance(index, tirx.FloorDiv):
                    if not (
                        structural_equal(index.a, store_base)
                        and _const_int(index.b) == row_len
                    ):
                        return None
                elif not structural_equal(index, stmt.loop_var):
                    return None
                if expr.buffer.dtype != dtype or not is_private(expr.buffer):
                    return None
                return element_ptr(
                    expr.buffer,
                    tirx.floordiv(chunk_origin, tirx.IntImm("int32", row_len)),
                    expand_rows,
                    1,
                )

            def cycle_operand(expr):
                """Access pointer for a ``buf[flat % period]`` cycle load."""
                if cycle_period is None:
                    return None
                if not isinstance(expr, tirx.BufferLoad) or len(expr.indices) != 1:
                    return None
                index = expr.indices[0]
                if not (
                    isinstance(index, tirx.FloorMod)
                    and structural_equal(index.a, store_base)
                    and _const_int(index.b) == cycle_period
                ):
                    return None
                if expr.buffer.dtype != dtype or not is_private(expr.buffer):
                    return None
                return element_ptr(expr.buffer, 0, cycle_period, 1)

            call_row_dep: list[bool] = []

            def mentions_row(expr) -> bool:
                if row_var is None:
                    return False
                found = [False]

                def visit(node) -> None:
                    if isinstance(node, tirx.Var) and node.same_as(row_var):
                        found[0] = True

                post_order_visit(expr, visit)
                return found[0]

            def is_access_ptr(expr) -> bool:
                return (
                    isinstance(expr, tirx.Call)
                    and str(getattr(expr.op, "name", "")).endswith("tvm_access_ptr")
                )

            def call_depends_on_row(args) -> bool:
                # Pointer arguments carry the per-row sliding window (base +
                # row * period) which is uniform for every row; only a value
                # argument such as a broadcast scalar makes the op row-specific.
                return any(
                    mentions_row(a) for a in args if not is_access_ptr(a)
                )

            def emit(name: str, *args) -> None:
                if os.environ.get("TL_MLU_DEBUG_MATCH"):
                    import sys
                    print(f"[emit] {store.buffer.name} {name} count={emit_count}", file=sys.stderr)
                stmt = tirx.Evaluate(
                    tirx.call_extern(
                        "void",
                        f"__tl_mlu_ew_{suffix}_{name}",
                        *args,
                        tirx.IntImm("int32", emit_count),
                    )
                )
                call_row_dep.append(call_depends_on_row(args))
                calls.append(stmt)

            def emit_bang(name: str, *args) -> None:
                if os.environ.get("TL_MLU_DEBUG_MATCH"):
                    import sys
                    print(f"[emitb] {store.buffer.name} {name}", file=sys.stderr)
                stmt = tirx.Evaluate(tirx.call_extern("void", name, *args))
                call_row_dep.append(call_depends_on_row(args))
                calls.append(stmt)

            def loop_invariant(expr) -> bool:
                if uses_var(expr, stmt.loop_var):
                    return False
                return inner is None or not uses_var(expr, inner.loop_var)

            def materialize(expr):
                if (
                    isinstance(expr, tirx.Cast)
                    and str(dtype) == "bfloat16"
                    and str(expr.dtype) == "bfloat16"
                    and str(getattr(expr.value, "dtype", "")) == "float32"
                ):
                    src = vector_float(expr.value)
                    if src is None:
                        return None
                    emit_bang(
                        "__bang_float2bfloat16",
                        dst,
                        src,
                        tirx.IntImm("int32", emit_count),
                    )
                    return dst
                source = vector(expr)
                if source is not None:
                    return source
                if isinstance(expr, tirx.Cast) and str(dtype) == "bfloat16":
                    return materialize(expr.value)
                if isinstance(expr, tirx.Call) and str(getattr(expr.op, "name", "")).endswith("exp2"):
                    if len(expr.args) != 1:
                        return None
                    # Triton-style chain fusion (mtp_592): the online-softmax
                    # panel update exp2((panel - cycle(row)) * scalar) is one
                    # NFU instruction, __bang_cycle_sub_mul_exp, instead of
                    # the cycle_sub + mul_scalar + pow2 triple.
                    arg = expr.args[0]
                    fused = None
                    mul_a = mul_b = None
                    if isinstance(arg, tirx.Mul):
                        mul_a, mul_b = arg.a, arg.b
                        if scalar(mul_a) is not None and isinstance(mul_b, tirx.Sub):
                            mul_a, mul_b = mul_b, mul_a
                    if (
                        mul_a is not None
                        and row_var is None
                        and scalar(mul_b) is not None
                        and isinstance(mul_a, tirx.Sub)
                        and cycle_operand(mul_a.b) is not None
                        and vector(mul_a.a) is not None
                    ):
                            fused = (
                                dst,
                                vector(mul_a.a),
                                cycle_operand(mul_a.b),
                                scalar(mul_b),
                            )
                    if fused is not None:
                        emit_bang(
                            "__bang_cycle_sub_mul_exp",
                            *fused,
                            tirx.IntImm("int32", emit_count),
                            tirx.IntImm("int32", cycle_period),
                        )
                        return dst
                    source = materialize(expr.args[0])
                    if source is None:
                        return None
                    emit("exp2", dst, source)
                    return dst
                op_name = {
                    "Add": "add",
                    "Sub": "sub",
                    "Mul": "mul",
                    "Max": "max",
                    "Min": "min",
                }.get(type(expr).__name__)
                if op_name is None:
                    return None
                lhs = materialize(expr.a)
                if lhs is None:
                    return None
                if os.environ.get("TL_MLU_DEBUG_MATCH"):
                    import sys
                    print(
                        f"[mat] {store.buffer.name} {op_name} scalar={scalar(expr.b) is not None} "
                        f"row={row_operand(expr.b) is not None} expand_rows={expand_rows} "
                        f"b={expr.b}",
                        file=sys.stderr,
                    )
                rhs = vector(expr.b)
                if rhs is not None:
                    emit(op_name, dst, lhs, rhs)
                    return dst
                rhs = scalar(expr.b)
                if rhs is not None and op_name in ("add", "sub", "mul", "max", "min"):
                    emit(f"{op_name}_scalar", dst, lhs, rhs)
                    return dst
                rhs = row_operand(expr.b)
                if rhs is not None and op_name in ("add", "sub", "mul", "max"):
                    bang = {"add": "add", "sub": "sub", "mul": "mul", "max": "maximum"}[op_name]
                    if os.environ.get("TL_MLU_BCAST_OP") == "cycle":
                        # __bang_expand_* needs arch >= 613; on mtp_592 fall
                        # back to the cycle variant (semantics under test).
                        emit_bang(
                            f"__bang_cycle_{bang}",
                            dst,
                            lhs,
                            rhs,
                            tirx.IntImm("int32", emit_count),
                            tirx.IntImm("int32", expand_rows),
                        )
                    else:
                        emit_bang(
                            f"__bang_expand_{bang}",
                            dst,
                            lhs,
                            rhs,
                            tirx.IntImm("int32", expand_rows),
                            tirx.IntImm("int32", row_len),
                        )
                    return dst
                rhs = cycle_operand(expr.b)
                if rhs is not None and op_name in ("add", "sub", "mul", "max"):
                    bang = {"add": "add", "sub": "sub", "mul": "mul", "max": "maximum"}[op_name]
                    emit_bang(
                        f"__bang_cycle_{bang}",
                        dst,
                        lhs,
                        rhs,
                        tirx.IntImm("int32", emit_count),
                        tirx.IntImm("int32", cycle_period),
                    )
                    return dst
                return None

            value = scalar(store.value)
            copied = vector(store.value)
            if value is None and copied is None and loop_invariant(store.value):
                value = tirx.Cast(dtype, store.value) if str(dtype) == "bfloat16" else store.value
            if value is not None:
                emit("fill", dst, value)
            elif copied is not None:
                emit_bang(
                    "__bang_move",
                    dst,
                    copied,
                    tirx.IntImm("int32", emit_count * dtype.bits // 8),
                )
            elif materialize(store.value) is None or not calls:
                return None
            result = calls[0] if len(calls) == 1 else tirx.SeqStmt(calls)
            if row_var is None:
                return result
            # Hoist the broadcast-independent tail of the chain out of the row
            # loop. Ops after the last row-dependent call touch only dst and
            # loop-invariant scalars, so one whole-panel call (count scaled by
            # the row count, offsets pinned at the panel start) replaces the
            # per-row copies. The row loop's vector ops and the hoisted calls
            # share the compute queue, so program order keeps the dependence.
            from tvm.tirx.stmt_functor import substitute

            tail_start = len(calls)
            for idx in range(len(calls) - 1, -1, -1):
                if call_row_dep[idx]:
                    tail_start = idx + 1
                    break
            if tail_start < len(calls):
                rows_n = count // period
                head = calls[:tail_start]
                tail = []
                hoistable = True
                for stmt in calls[tail_start:]:
                    call = stmt.value
                    if not isinstance(call, tirx.Call):
                        hoistable = False
                        break
                    args = list(call.args)
                    amount = _const_int(args[-1])
                    if amount is None:
                        hoistable = False
                        break
                    zero = tirx.IntImm("int32", 0)
                    new_args = [
                        substitute(a, {row_var: zero}) for a in args[1:-1]
                    ]
                    new_args.append(tirx.IntImm("int32", amount * rows_n))
                    tail.append(
                        tirx.Evaluate(tirx.call_extern("void", args[0], *new_args))
                    )
                if hoistable:
                    if not head:
                        return tail[0] if len(tail) == 1 else tirx.SeqStmt(tail)
                    loop_body = head[0] if len(head) == 1 else tirx.SeqStmt(head)
                    row_loop = tirx.For(
                        row_var,
                        tirx.IntImm("int32", 0),
                        tirx.IntImm("int32", count // period),
                        tirx.ForKind.SERIAL,
                        loop_body,
                    )
                    return row_loop if not tail else tirx.SeqStmt([row_loop] + tail)
            return tirx.For(
                row_var,
                tirx.IntImm("int32", 0),
                tirx.IntImm("int32", count // period),
                tirx.ForKind.SERIAL,
                result,
            )

        # Pre-order matching: the OUTERMOST loop wins, so a whole-panel copy
        # lowers to one large memcpy instead of being fragmented into
        # per-inner-row copies by a bottom-up visit. Patterns that only match
        # at an inner level (e.g. single-row copies) still match when the
        # walk descends past an unmatched outer loop.
        return func.with_body(ir_transform(func.body, lower, None))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.LowerMLUElementwise")


def LowerClusterStaging():
    """Lower tl.mlu_stage_load markers into the cluster SRAM protocol.

    See tilelang/mlu/transform/cluster_staging.py for the producer/consumer
    structure the pass injects (the script never mentions the MPU)."""
    from . import cluster_staging

    def pass_fn(func, mod, ctx):
        from . import cluster_staging

        if not cluster_staging.has_shared_fill(func.body):
            return func
        # IPU count from the MLU target attrs (src/mlu/target.cc):
        # cluster_dim x core_dim.
        ipus = cluster_staging._IPUS_PER_WAVE_DEFAULT
        target = func.attrs.get("target")
        if target is not None:
            try:
                clusters = int(target.attrs["cluster_dim"])
                cores = int(target.attrs["core_dim"])
                if clusters > 0 and cores > 0:
                    ipus = clusters * cores
            except (KeyError, TypeError, ValueError):
                pass

        def ir_transform(stmt, None_, lower_):
            return cluster_staging.lower(stmt, ipus)

        return func.with_body(ir_transform(func.body, None, None))

    return prim_func_pass(pass_fn, opt_level=0, name="tl.LowerClusterStaging")
