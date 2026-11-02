# TileLang MLU590 Backend

MLU590 is a TileLang target backend. BANG C and CNRT are the device language
and the launch API. [mlu-ops](https://github.com/Cambricon/mlu-ops) is the
reference for instructions, alignment, and launch classes. It is not linked
into the compiler.

The ownership model is
[`tilelang/backend/README.md`](../backend/README.md). New work follows
[`.agents/skills/tilelang-backend/SKILL.md`](../../.agents/skills/tilelang-backend/SKILL.md)
and its integration checklist. This page is the capability contract for that
checklist. The closest pipeline reference is CPU: the launch is a grid of
programs, and there is no `threadIdx`.

## Preflight

| Choice | Value |
| --- | --- |
| Package | `tilelang/mlu` |
| Native tree | `src/mlu` |
| Target kind | `mlu`, device type `kDLExtDev`. TileLang registers it. TVM does not. |
| Aliases | `mlu`, `mlu590`, `mtp_592` |
| Attributes | `mcpu=mtp_592`, `arch=mtp_592`, `bang_arch=592`, `core_dim=4`, `cluster_dim=8`, `max_cluster_dim=16`, `nram_bytes=524288`, `wram_bytes=524288`, `sram_bytes=2097152`, `nram_stack_reserve=131072`, `nfu_align_bytes=128`, `wram_align_bytes=64`, `compute_count_align=64` |
| Flags | `supports_kernel_status_return=false`, `launch_grid_only=true` |
| Auto-detect | `torch.mlu.is_available()` only. A CNToolkit install does not claim the host. The package is imported before CUDA. |
| Variant predicate | none. One manifest owns `mlu`. |
| Dialect | common TileLang language. `threads` and `cluster_dims` are rejected. |
| v1 tile ops | scalar global loads and stores; one 64-aligned vector loop lowered to NRAM and `__bang_add` / `__bang_add_scalar` |
| Launch | Scalar Block: `blockIdx.x` is `taskIdX`, `dim` is the TileLang grid. Cluster GEMM: `blockIdx.x` is `taskIdX / 4`, Union1, `dim.x = grid.x * 4`. |
| Device output | BANG C, then `cncc -c` object bytes via `tilelang_callback_mlu_compile` |
| Host | `STANDARD_HOST_CODEGENS` (`c` / `llvm`) |
| Execution | `tvm_ffi`, host codegen on, eager device compile on |
| Hardware-free checks | target normalization, pipeline IR, generated BANG C, `cncc -c` where `cncc` exists |
| Hardware checks | CNRT launch on MLU590-M9DG |

`max_cluster_dim=16` is the `cnrtFuncTypeUnion16` job class. MLU590-M9DG has 8
clusters and 4 IPUs per cluster. Usable NRAM is `nram_bytes - nram_stack_reserve`
(384KiB), matching mlu-ops `MAX_NRAM_SIZE`.

## Memory and launch

Measured on MLU590-M9DG with `cnDeviceGetAttribute`: isa 592, 8 clusters, 4
IPUs, NRAM 512KiB, WRAM 512KiB, SRAM 2MiB. mlu-ops reserves 128KiB of NRAM for
the cncc stack (`REM_FOR_STACK`). `__bang_*` counts are multiples of 64
elements. NFU chunks are 128 bytes. WRAM alignment is 64 bytes.

| TileLang | BANG | Owner | v1 |
| --- | --- | --- | --- |
| Global buffer | GDRAM pointer argument | device | direct indexing |
| `T.Kernel` program index | `blockIdx.x` | one logical cluster, or one Block task | yes |
| `threads=` | none. `coreId` is not a host-chosen thread id | — | reject |
| Fragment / vector tile | NRAM, `__nram__` | one IPU | 64-aligned vector loop |
| `alloc_shared` / `shared.dyn` | SRAM, `__mlu_shared__` | one cluster | cluster GEMM helper |
| GEMM weight tile | WRAM, `__wram__` | one IPU | inside the cluster tile helper |

## Cluster program

One `T.Kernel` iteration is one logical cluster: 4 IPUs and 1 MPU. The
argument is the grid of those programs, the same role as a CUDA grid. It is
not the physical cluster count (8 on MLU590-M9DG) and not a task count.

```python
with T.Kernel(T.ceildiv(M, TILE_M)) as bx:
    m0 = bx * TILE_M
    T.gemm(A[m0 : m0 + TILE_M, :], B, C[m0 : m0 + TILE_M, :], clear_accum=True)
```

`bx` is the only index the kernel sees. `examples/mlu/example_gemm.py` is
this launch. `TILE_M` is a multiple of 16 and at most 256, so one cluster's
NRAM can hold its rows. N and K of that row tile are still walked inside
the BANG helper. `alloc_shared`, `T.copy`, and `T.Pipelined` are the next
step; they are how a kernel will schedule the K panels itself.

CNRT assigns the physical cluster. The host passes `dim` and
`cnrtFuncTypeUnion1`, not a cluster id. The device builtin `clusterId` is
the slot inside the union. Under Union1 that slot is 0 on every running
cluster, so it is not `bx`.

The launch expands one logical cluster into four IPU tasks:

```text
dim.x  = grid.x * 4
bx     = taskIdX / 4
core   = taskIdX % 4
```

`taskId` stays in the runtime and in those two expressions. Codegen prints
`blockIdx.x` as `(taskIdX / 4)`. `blockIdx.y` and `blockIdx.z` print as
`taskIdY` and `taskIdZ`. The tile helper splits the tile's rows with
`row_begin = (taskIdX % 4) * (M / 4)`. The MPU takes the `__is_mpu()` branch
and copies the shared B panel GDRAM to SRAM. The four IPUs compute.

Union1 is the job class for this grouping: `dim.x` is a multiple of 4, and
`__sync_cluster` covers one cluster. Union2/4/8 need `dim.x` to be a multiple
of 8/16/32 when a later kernel must sync across clusters. This card's full
co-scheduled launch is Union8. Union16 stays unused.

Scalar v1 stays `cnrtFuncTypeBlock`. There `dim` equals the TileLang grid and
`blockIdx.x` is `taskIdX`. Block does not share SRAM, so that grid need not
be a multiple of 4. `examples/mlu/vector_add.py` is this launch.

## What already runs

- `src/mlu/target.cc` registers the kind. `src/mlu/CMakeLists.txt` compiles it
  in every build. The top-level `CMakeLists.txt` includes that file.
- `tilelang/mlu/__init__.py` normalizes the three aliases.
- `tilelang/mlu/backend.py` registers the manifest. The compile callback runs
  `cncc -c -fPIC -O3 -std=c++11 --bang-mlu-arch=<mtp_N>`. The pipeline and
  device codegen are wired. Cluster GEMM lowers one row tile per logical
  cluster.
- `examples/mlu/vector_add.mlu` copies GDRAM to NRAM, adds one with
  `__bang_add_scalar`, and copies back. On MLU590-M9DG it printed
  `PASS mismatches=0`.
- A pod build at `/tmp/tilelang-build` produced `libtilelang.so`.
  `determine_target("mlu")` returns the attributes above, and the callback
  emitted a 5664-byte object. The checkout's `build` entry is a symlink to
  that directory.

The lsyncd mirror does not delete files. A removed source that is still
present on the pod will be compiled again. `src/target_mlu.cc` did this once
and registered `mlu` twice (`Duplicate config option key: 'kind'`). Delete the
remote copy by hand.

## Compiler flow

`BackendModule` already requires one pipeline per kind, one device codegen,
one execution backend, and host codegen because `tvm_ffi` enables it. The
manifest has that shape. The function bodies are the remaining work.

```text
common frontend IR
        |
        v
PassPipeline("mlu")          tilelang/mlu/pipeline.py
        |
        v
host / device split
        |
        +--> STANDARD_HOST_CODEGENS
        |
        +--> DeviceCodegen               tilelang/mlu/codegen.py
                 |
                 v
             BANG C                       src/mlu/codegen/codegen_bang.cc
                 |
                 v
             tilelang_callback_mlu_compile    cncc -c -> .o
                 |
                 v
             MLU runtime module           src/mlu/runtime/mlu_module.cc
                 |
                 v
             tvm_ffi PackedFunc
```

The `.o` from `cncc -c` is not a CNRT-loadable module. The runtime slice reads
`cnrt.h` on the pod first, then picks one loader: link the object with a small
host stub into a `.so` and `dlopen` it, or call a CNRT API that accepts a BANG
object. Launch dimensions stay in that runtime module. Host C only packs
arguments and calls the device function.

`TVM_FFI_BUILD_PYTHON_MODULE` returns early when tvm-ffi is a subdirectory, so
the Python package is `apache-tvm-ffi>=0.1.11,<0.1.13`, not a CMake option.

## Pipeline

`tilelang/mlu/pipeline.py` follows the CPU pass order and omits
`LowerCPUKernelLaunch`, `MarkCPUAtomics`, `LowerCPUAtomics`, and
`MaterializeCPUParallelGrid`.

The MLU-specific launch step is:

```text
MaterializeKernelLaunch(lower_grid_binding=True, lower_thread_binding=False)
```

`threads` and `cluster_dims` fail there with an actionable error. The rest of
the order is the shared legalization: let-inline, negative indices, assumes,
Simplify, reducers, `VerifyBufferInit`, `LayoutInference`, `LowerTileOp`,
pointer and dtype legalization, allocation and `StorageRewrite`,
`SplitHostDevice`, `MakePackedAPI`. If the codegen has no BF16 type, keep
`BF16ComputeLegalize` after `FlattenBuffer`, as the CPU pipeline does.

`LowerTileOp` lowers `T.gemm` to `tl.mlu_bang_gemm` on the cluster tile.
A specialized reduce does not become a silent scalar loop.

## Pipelining pattern (measured)

Flash attention (`examples/mlu/flash_attention.py`) is the reference. Its
design, queue/synchronization contract, and the full optimization ledger
(what worked, what was falsified, and the methods) live in
[`flash_attention_design.md`](flash_attention_design.md). The
manual form beats and replaces `T.Pipelined` on this backend: same
performance, and the generated BANG C mirrors the source one-to-one.

```python
from tilelang.mlu.language import sync           # the MLU dialect, not the shared facade

k = T.alloc_fragment((2, block_n, dim), dtype)   # explicit ring, 2 slots
v = T.alloc_fragment((2, block_n, dim), dtype)
T.copy(K[..., :block_n, :], k[0, :, :])
T.copy(V[..., :block_n, :], v[0, :, :])
sync("io")
for ko in T.serial(T.ceildiv(seq_kv, block_n)):
    sync("io")                       # wait the DMA issued last iteration
    if ko + 1 < T.ceildiv(seq_kv, block_n):
        T.copy(K[..., next tile, :], k[(ko + 1) % 2, :, :])   # issued AFTER the
        T.copy(V[..., next tile, :], v[(ko + 1) % 2, :, :])   # sync, so its DMA
        # overlaps the compute below; the next sync("io") waits it
    T.gemm(q, k[ko % 2, :, :], score, ...)
    ... softmax ...
    T.gemm(score_bf, v[ko % 2, :, :], output, ...)
```

Why it is safe with one `sync("io")` per iteration: `T.gemm` internally
drains the compute queues (`__sync_compute` before reusing its WRAM filter)
and copies the filter with a synchronous `__memcpy`, so a ring slot is dead
once the consuming `T.gemm` returns. The next iteration's async copy into
that slot cannot race any compute read. `T.sync("move")` and
`T.sync("compute")` exist for the interface but belong inside `T.gemm`.

The lowering does the rest automatically:

| Form | Lowering |
| --- | --- |
| Copy into a ring slot (buffer numel > tile, load direction) | `__memcpy_async` |
| Dense GEMM operands (`N*K <= 64K` elements) | one `__bang_reshape_filter(..., NRAM2WRAM)` + one `__bang_matmul` per call; oversized tiles fall back to 64x64 tiling |
| Row-broadcast softmax chains (`row_max[(i*c+j)//c]`) | one vector chain per NFU row |
| Whole-panel `mul_scalar` / `exp2` | one call per panel |

Measured dead ends on mtp_592, do not retry without new evidence:
`__bang_expand_*` (needs BANG arch >= 613), `__bang_cycle_*` as a row
broadcast on a row-major panel (cycles by element along the fast axis; it is
correct only on a transposed panel, and the two per-iteration transposes
cost more than the row chain they replace at long contexts), pool-based
2D reduces (`__bang_maxpool`/`__bang_sumpool` lose to per-row
`__bang_max`), and `T.Pipelined` (its implicit versioning and barrier
insertion reproduce the manual form with more machinery and less
visibility).

## Codegen

`CodeGenBANG` subclasses `CodeGenTileLangCBase`
(`src/backend/common/codegen/codegen_c_base.h`). It emits
`__mlu_global__ void` kernels. v1 accepts loops, branches, scalar arithmetic,
and global `BufferLoad` / `BufferStore`. A vector loop whose trip count is a
multiple of 64 becomes an `__nram__` buffer, `GDRAM2NRAM`, `__bang_add` or
`__bang_add_scalar`, and `NRAM2GDRAM`.

These forms fail at the statement that contains them: a user-level `shared`
allocation, atomics, CUDA or Ascend intrinsics, and `threadIdx`. The cluster
GEMM helper is the one place that emits SRAM, WRAM, and `__bang_matmul`.
Register both
`target.build.tilelang_mlu` and `target.build.tilelang_mlu_without_compile`.
The source-only entry skips `cncc`.

## Work slices

One file per slice, at most three functions.

| Step | File | Done when |
| --- | --- | --- |
| 1 | `tilelang/mlu/pipeline.py` | A scalar primfunc lowers. The IR has a launch index and no `threadIdx`. |
| 2 | `tilelang/mlu/backend.py` | The manifest pipeline is step 1. Device codegen still refuses. |
| 3 | `src/mlu/codegen/codegen_bang.cc` | Scalar GDRAM source passes through the existing `cncc` callback and yields a non-empty object. Also add the source to `src/mlu/CMakeLists.txt` in that same native slice. |
| 4 | `tilelang/mlu/codegen.py` | `DeviceCodegen` calls `target.build.tilelang_mlu`. Point `backend.py` at it on the next edit if this slice is already one file. |
| 5 | `src/mlu/runtime/mlu_module.cc` | After reading `cnrt.h`, a Block launch of scalar "add one" matches the host. |
| 6 | one function in `codegen_bang.cc` | The 64-aligned NRAM loop matches `examples/mlu/vector_add.mlu` (`mismatches=0`). |
| 7 | SRAM lowering | `alloc_shared` uses SRAM and a Union1 or Union8 launch. Two IPUs exchange one tile. |

Dialect file `tilelang/mlu/language/__init__.py` is the common re-export from
the checklist. Add it when the first example imports `tilelang.mlu.language`.
Until then, examples use `tilelang.language` and the pipeline rejects CUDA-only
launch annotations.

## Build and check

On the MLU pod, build out of tree so objects stay off the sync mirror:

```bash
cmake -S <tilelang> -B /tmp/tilelang-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_ASCEND=OFF
ninja -C /tmp/tilelang-build
```

The pod's system CMake is 3.24. TileLang requires 3.26 or newer (`pip install
'cmake>=3.26'`). Cython is required for the wrapper targets. Z3 comes from
`z3-solver`. There is no `nvcc` and no `llvm-config`.

```bash
cd <tilelang>
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH=/tmp/tilelang-build/lib:$NEUWARE_HOME/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD
python -c "import tilelang; from tilelang.backend.target import determine_target; print(determine_target('mlu', return_object=True))"
```

After step 5, `tilelang.compile(..., target="mlu")` on a one-axis `T.Kernel`
that adds one to a global buffer matches the host. After step 6, the same
shape matches `examples/mlu/vector_add.mlu`.

## v1 boundary

v1 scalar kernels are `mtp_592`, Block launch, scalar GDRAM, and one NRAM
vector loop. Cluster GEMM is a Union1 launch of logical clusters; the helper
uses SRAM, WRAM, and `__bang_matmul` on one row tile. CUDA `threadIdx`
lowering and the Ascend UB pipeline are not used.
