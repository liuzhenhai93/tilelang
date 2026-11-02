# MLU590 存储层次与同步

对象是 MLU590-M9DG。数字来自该卡上的 `cnDeviceGetAttribute`，语义来自 mlu-ops 的 [BANG C 开发指南](https://github.com/Cambricon/mlu-ops/blob/master/docs/BANG-C-OPS-Develop-Guide.md) 和仓库里实际调用的 builtin。指南里的 MLU Core 在这里写成 IPU，Memory Core 写成 MPU。

## 芯片怎么切

一张卡 8 个 cluster。每个 cluster 是 4 个 IPU 加 1 个 MPU。

IPU 是计算核，有自己的 NRAM 和 WRAM。MPU 是这个 cluster 的 DMA 和控制核，没有 NRAM / WRAM。4 个 IPU 和这 1 个 MPU 共用本 cluster 的 SRAM。IPU 读不到另一个 IPU 的 NRAM。cluster 之间不共用 SRAM，跨 cluster 只经过 GDRAM。

CNRT 负责把任务放到空闲的物理 cluster 上。设备侧的 `clusterId` 是这次 Union 任务内部的槽位。Union1 下这个槽位恒为 0，它不是物理 cluster 编号，也不是 host 上的 launch 索引。

## 存储

| 存储 | MLU590-M9DG | 谁能访问 | 放什么 |
| --- | --- | --- | --- |
| GPR | 每核一组标量寄存器 | 该 IPU 或该 MPU | 地址、循环变量、标量运算 |
| NRAM | 每 IPU 512KiB，可用 384KiB | 该 IPU | 向量和张量的输入、输出、临时数据 |
| WRAM | 每 IPU 512KiB | 该 IPU | 卷积 / matmul 的 filter，布局由硬件规定 |
| SRAM | 每 cluster 2MiB | 本 cluster 的 4 IPU + MPU | 本 cluster 内共享的 tile |
| GDRAM | 设备全局内存 | 全部 IPU 和 MPU | kernel 输入输出，以及跨 cluster 交换 |
| LDRAM | 每核一块片外私有空间，容量由软件配置 | 该核 | 片上放不下的私有数据 |
| L2 Cache | 芯片共享 | 硬件管理 | 指令、kernel 参数、只读数据 |

NRAM 的 512KiB 里，cncc 栈占 128KiB（mlu-ops 的 `REM_FOR_STACK`）。用户缓冲按 384KiB 预算。

NRAM 是计算的工作集。向量指令要求 128 字节对齐，元素个数是 64 的倍数。WRAM 对齐 64 字节。matmul 的 A 在 NRAM，filter 在 WRAM。WRAM 不拿来当普通缓冲。

SRAM 只在 Union 任务里可用。Block 任务没有 SRAM，也不能做 cluster 间通信。一块 tile 要给本 cluster 的 4 个 IPU 一起读时才进 SRAM，例如 GEMM 的 B、或者 IPU 之间的归约槽。归约的做法是各 IPU 把部分结果写到 `sram[coreId]`，IPU 之间同步，再把 4 个槽读回自己的 NRAM。被归约的那一维没有拆到多个 IPU 上时，结果留在 NRAM。

GDRAM 是唯一有原子操作的一层。mlu-ops 里看到的是 `__bang_atomic_reduce_add` / `max` / `min`：源在 NRAM，目的在 GDRAM，走 IO 队列，浮点累加不保证顺序。调用它们的 context 必须是 `MLUOP_ATOMICS_ALLOWED`；默认是 `MLUOP_ATOMICS_NOT_ALLOWED`。mlu-ops 的设备 kernel 里没有 SRAM 原子，也没有 cas / exch / and / or / xor。

LDRAM 和 L2 在编程模型里存在。TileLang 的 kernel 使用 GDRAM、SRAM、NRAM、WRAM 这四层。

## 一个 IPU 上的队列

指令先按顺序译码，再进不同队列。MLUv03 上片外 DMA 和片上 DMA 互相独立。GPR 上的标量依赖由硬件寄存器重命名处理，用户不等它。

用户要自己等的是三条队列：

| 队列 | 指令落在哪 | 典型操作 |
| --- | --- | --- |
| IO | `__sync_io` | IPU 的 GDRAM↔NRAM。MPU 的 GDRAM↔SRAM 也是这条片外 DMA |
| Move | `__sync_move` | SRAM↔NRAM、NRAM↔WRAM、`__bang_move`、`__bang_transpose`、类型转换 |
| Compute | `__sync_compute` | 向量运算和 matmul。向量单元和张量单元对程序员是这一条计算队列 |

算术 elementwise 在 Compute，不在 Move。标量除法、乘法如果没进向量单元，留在按顺序执行的标量流水上。

这三条同步都是排空整条队列，不是等某一个 tile。`__sync_io()` 会等掉该核上还没完成的全部 IO。因此同一条 IO 队列上通常只保持一份在飞的拷贝；要重叠，就给被重叠的那一块缓冲准备两个槽。

无参 `__sync_io_move_compute()` 让这条调用之后的 IO、Move、Compute 指令，等到调用之前这三条队列里的指令全部完成。六参数形式是 `(ioproducer, moveproducer, computeproducer, ioconsumer, moveconsumer, computeconsumer)`，默认全为 `true`。消费者队列里、调用之后的指令，等到生产者队列里、调用之前的指令完成。生产者和消费者都不能全是 `false`。mlu-ops 的 MSDA 用过 `(true, false, false, false, true, false)`，即 Move 等之前的 IO。

## Cluster 和 Union 的屏障

| Builtin | 谁必须到达 | 范围 |
| --- | --- | --- |
| `__sync_cluster()` | 本 cluster 的 MPU 和 4 个 IPU | 一个物理 cluster。指令是 `barrier.sync.local`，计数 `(%coreDim + 1)`。Union2 及以上也仍然只罩这一个物理 cluster |
| `__sync_all_ipu()`、`__sync_all_ipu_within_cluster()` | 本 cluster 的 IPU | 不含 MPU。把 MPU 算进这个屏障会死锁 |
| `__sync_all()` | 这次 UnionN 任务里的全部 cluster | `barrier.sync.global`，计数 `(%coreDim + 1) * %clusterDim` |

`__sync_cluster` 必须写在所有参与者都能执行到的位置。放在只有 MPU 或只有部分 IPU 会走进去的 `if` 里，其余核到不了，屏障等死。mlu-ops 的 `samples/mlu-ops/fault_sample/fault_kernel.mlu` 就是 MPU 单独同步死锁的例子。

## 任务类型决定能用哪层同步

任务类型决定一个 kernel 一次占用多少 cluster。MLU590 有 8 个 cluster，能发 Union1 / 2 / 4 / 8。`cnrtFuncTypeUnion16` 是接口里的任务类，这张卡排不下。

| 类型 | `dim.x` | SRAM | cluster 之间 |
| --- | --- | --- | --- |
| Block | 任务数即 IPU 数 | 不可用 | 不能通信。`blockIdx.x` 就是 `taskIdX` |
| Union1 | 4 的倍数 | 这一个 cluster 的 4 IPU + MPU 共享 | 不能通信。一个任务占 1 个 cluster |
| Union2 / 4 / 8 | 8 / 16 / 32 的倍数 | 每个 cluster 仍是自己的 2MiB | 可以通信，走 GDRAM 加 `__sync_all` |

Union 后面的数字是这次任务占用的 cluster 个数，不是“芯片上只用一个 cluster”。Union1 任务可以被调度到 8 个物理 cluster 上排队。整机任务类型一致时，调度器看到的粒度统一。Union2 要等两个 cluster 同时空闲。

UnionN 的额外能力是把 N 个 cluster 一起调度，并用 `__sync_all` 在它们之间做 GDRAM 交换。它不把这 N 个 cluster 的 SRAM 拼成一块。

## 数据怎么走

常见路径不经过 SRAM：IPU 用 IO 队列做 GDRAM→NRAM，在 NRAM 上计算，再 NRAM→GDRAM。三级流水把 NRAM 分成两槽，让下一块的 Load 叠在当前块的 Compute 上。Load 和 Store 都在 IO 队列里，按发射顺序执行，所以两槽要错开，避免同一槽被还没写完的 Store 和下一次 Load 同时使用。

四级流水只把 Load 拆开：MPU 做 GDRAM→SRAM，IPU 做 SRAM→NRAM，结果仍由 IPU 从 NRAM 写回 GDRAM。五级流水把 Store 也拆开，结果先 NRAM→SRAM，再由 MPU 写回 GDRAM。SRAM 上要做这种重叠时同样是两槽。一槽的话，IPU 还在读时 MPU 不能覆盖，每个 tile 需要两次 cluster 屏障。

IPU 发往 GDRAM 的 DMA 目的地是 NRAM。GDRAM→SRAM 由 MPU 发出。有的 unary kernel 在公共指令流上发这条拷贝，严格的写法是 `__is_mpu()` 里发。

## TileLang 里用户看得到的部分

`T.Kernel(...)` 的网格是逻辑 tile。物理 cluster 由 CNRT 分配。

fragment 在 NRAM。`alloc_shared` 在 SRAM。WRAM 不作为用户缓冲；`T.gemm` 自己把 filter 放进 WRAM，并在复用前插入它需要的 `__sync_compute` / `__sync_move`。

`T.sync` 对应四条 builtin：`"io"`、`"move"`、`"compute"`、`"cluster"`。用户编排 IO 和 cluster。`move` 留给用户自己发出的 Move。`compute` 留给 `T.gemm` 内部。
