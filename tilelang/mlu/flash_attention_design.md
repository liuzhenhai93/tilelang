# MLU590 FlashAttention 内核：设计、同步机制与优化经验

交付物：`examples/mlu/flash_attention.py`（统一入口：conv 主路径 + gemm 兜底）。
成绩（稳健默认路径）：2048/8192/16384 = **0.082 / 0.371 / 1.349 ms**（13.2/46.3/51.0 TF）。
基线对照：`flash_attention_gemm.py`（旧 `__bang_matmul` 主线，0.096/0.790/2.963）。

对 cnnl FA2 的同场对比（同 pod、同张量、同协议，`bench_vs_fa2.py`；
fast = `softmax_shift=10` 路径——**ln2 bug 修复后数值已对齐** 0.0005）：

| seq | ours safe | ours fast | FA2 (cnnl SDPA) | fast/FA2 |
|---|---|---|---|---|
| 2048 | 0.082 ms @13.0 TF | **0.075 ms @14.3 TF** | 0.166 ms @6.5 TF | **0.45（快 2.2×）** |
| 8192 | 0.374 ms @46.0 TF | **0.326 ms @52.7 TF** | 0.331 ms @51.8 TF | **0.98（反超）** |
| 16384 | 1.348 ms @51.0 TF | **1.164 ms @59.1 TF** | 1.229 ms @55.9 TF | **0.95（反超）** |

FA2 在小 seq 上只有 5.6 TF（launch/调度开销主导）——我们小形状反超 2.3×；
大 seq 靠 batched 调度（§1.4）+ f32-PV（§3.6）全面反超。两个实现的输出
逐元素差 0.0005（对齐 cnnl 的语义）。

## 1. 总体设计

### 1.1 执行模型

- **flat Union1 launch**：`taskDim = ceil(ntiles / tpt)`，`tpt = ceil(ntiles/32)`；
  Union1 要求 `taskDim ≡ 0 (mod 4)`（runtime 有显式报错）。每 task = 1 个 IPU 核。
- task `bx` 拥有 `tpt` 个连续 tile，`row0 = (bx*tpt + t) * block_m`。逻辑索引空间
  只覆盖 per-IPU 视图，无 split pass、无簇级 launch 形态。
- 三级片上存储（实测 `%__nram_size__` / `%__wram_size__` 均 512KB/核）：
  - **NRAM**（核私有）：Q/score/p/output 全部工作集。block_m=128 时 ~272KB。
  - **WRAM**（核私有）：conv filter 专用（`T.alloc_wram`）。**filter 基址必须
    < 32KB**（payload 可跨界）——同时最多驻留 2 个 16KB filter。
  - **SRAM**（簇共享 2MB）：本工具链（dlsym+cnrtInvokeKernel）下 **LSU 不可寻址、
    DMA 慢**——不可用（见 §3.3）。
- K/V 流量 = `(seq_q/block_m) × seq_kv × 32KB`。block_m=128（对齐 cnnl/Triton）
  比 64 减半；能开 128 的前提是 score/output/v0 **单槽化**省出的 NRAM。

### 1.2 conv 引擎（核心解码成果）

1×1 convolution = matmul，编码（差分 dump 法解出，全部探针验证）：

```
conv.nram.rn.f32.bf16.bf16 [dst],[in],[filter], ci, rows, 1,1,1,1,1, co,
    .mstride(1), .kstride(ci*2 bytes), .indilation(1,1), <融合修饰>
```

- `rows` = 本次调用行数（单调用覆盖全块）；**输出纯行主序 f32**。
- `.kstride` **接受寄存器** → `ci` 是自由运行时值（`__tl_mlu_conv_gen` 单 helper
  通吃）；`.mul.scalar.rn(s)`（QK 的 scale 融合，免一趟乘法）、
  `.add.partial.rn([dst])`（PV 原地累加，免读旧 C）、`.cvt.src.rn.f32()`
  （f32 输入专用——喂 bf16 会毒化数值）。
- filter = `(co, ci)` bf16 矩阵经 `__bang_reshape_filter(dst, src, co, 1, 1, ci)`
  打包（与 Triton `ld.3d`/`trans.tiling` 直产的布局一致）。
- 速率 **~31 TF/核**（背靠背流水）vs `__bang_matmul` ~1.2 TF/核（K=64）。
- 硬性限制：输出只能 f32（bf16 dst 被 cnas 0117 拒）。

### 1.3 增广 filter 与无 max softmax

- **增广 filter**：`wv = [V^T; ones]`（co = dim*2）。PV conv 的输出右半
  **就是行和**（Σp）——行归约随 matmul 免费完成，无需独立归约树。
  代价：PV FLOPs 翻倍——在 conv 速率下不重要。
- **无 max softmax**：`p = exp2(min(score·scale, 100))`。max 减法只是数值条件化；
  clamp 到 100 后 f32 累加和 < 2^127 永不溢出 → 删掉 max 跟踪/rescale/两趟转置。
  exp2 链必须留在 **f32**：bf16 链会被 exp2 放大 pre-exp2 舍入
  （大 logit 实测误差 0.186，超容差 9×）。

### 1.4 batched-overlap 调度（block_m=128 版，当前主线）

每 panel 迭代把"全部待办 MOVE 工作"合并成一段（stream 不冷启动、延迟只付
一次），K/V 用双缓冲槽位预取：

```
prologue: K(0)/V(0) sync 载入 → v0 转置 → wk/wv reshape → QK(0) → softmax(0)
          bootstrap: K(1)/V(1) sync 载入 + wk(1)；预取 K(2)/V(2)（槽 (2)%2）
迭代 ko:
  conv PV(ko)   [NFU]  ← 读 f32 score（.cvt.src，无 cast 一趟）
  conv QK(k+1)  [NFU]  ← 与 PV 背靠背，写单槽 score
  sync("io")           ← NFU 结果可读 + 先序 DMA 落地
  batch(k+1) [MOVE]:   转置 v0←v_t[(k+1)%2]；reshape wv；
                       reshape wk←kbuf[(k+2)%2]（服务下一迭代的 QK(k+2)）；
                       softmax(k+1)（min+exp2 或 shift 后纯 exp2）
  sync("move")
  DMA: K(k+3)→kbuf[(k+3)%2]；V(k+2)→v_t[(k+2)%2]   ← 同步拷贝，非 async
```

**槽位不变量**：K(j)/V(j) 驻留槽 `j%2`；batch(m) 的 wk reshape 服务
**QK(m+1)**（下一迭代发射），所以必须从槽 `(m+1)%2` 取——写成 `(m)%2` 是
off-by-one（QK 复用上一 panel 的 K filter，panel≥2 全错，panel-0/1 正常，
`dbg_panels.py` 的逐 panel 累加部分和二分定位）。wv 槽位跟随 `(m)%2` 是
对的（它服务同迭代的 PV(m)）。

**单槽正确性不变量**：score 被 QK(k+1) 覆盖前，PV(k) 已在 in-order NFU 里
先发射；wv 只在 PV(k) retired（sync("io") 隔开）后重填；panel 0 在
prologue 显式填充（循环只填 k+1——SRAM 版 race 的教训）。

### 1.5 epilogue

行倒数 `row_sum[i] = 1/output[i, dim]`（避免子视图 fdiv 标量化成 ~8192 次标量除）
→ **全宽** rescale（`output[i,j] *= row_sum[i]`，j 走满 dim*2——左半子视图会让
元素级 matcher 标量化，实测 11× 回退）→ 全宽 cast → 逐行 128B `T.copy` store
（`T.copy(out_full[r, 0:dim], O[bz,by,row0+r,:])`，行内连续、外层行号为不变量，
matcher 天然匹配）。

## 2. 同步机制

### 2.1 三条硬件队列

| 队列 | 承载 | 实测特性 |
|---|---|---|
| **DMA** | `__memcpy`/`__memcpy_async`（GDRAM↔NRAM/SRAM） | 延迟 0.67µs/16KB；async 与 sync 同延迟；同步 memcpy 隐式等**整个先序** DMA 队列 |
| **MOVE** | `__bang_*` 元素级/转置/`reshape_filter`/cast | **带宽 69GB/s/核**；单批往返延迟 ~2.7µs、批内流水 277GB/s——`sync("move")` 切小批=每批付全额延迟 |
| **NFU/计算** | `conv.*`/`__bang_matmul`/muladd | in-order；conv 背靠背深流水 124TF（聚合 4 task），被 sync 切断后按延迟计 |

### 2.2 sync 语义（实测口径）

- `sync("io")`：排空本核 **DMA** 队列；**同时使 NFU（conv/gemm）结果可读**
  （探针反复验证）。
- `sync("move")`：排空 MOVE 队列。
- `sync("cluster")`：簇级 barrier；其 `pio/cio` tag **自带全队列排空 + 写发布**
  （SRAM 协议的正确性钥匙）。MPU 上下文不执行内核 → 需 `LowerClusterStaging`
  波次匹配改写（MPU 循环 ×taskDim/ipus，ipus 从 target attrs 读
  `cluster_dim × core_dim`）。
- 裸 `call_extern("__sync_all")` 会被管线丢弃——必须用 `sync()` helper。
- **跨队序规则**：标量写→队列读须 `sync("move")`；队列写→标量读同理（epilogue 前）。

### 2.3 消费者契约

`T.copy` 的 MLU lowering（`try_memcpy` matcher）：ring/versioned 槽的加载自动
`__memcpy_async`；普通加载 sync（DMA 延迟在深度-1 调度下本就被 MOVE 掩盖，
实测两者性能相同）。**先序匹配**（外层大拷贝优先整体成一条 memcpy）是
2026-11-02 的修复——bottom-up 遍历会把 16KB 拷贝碎片化成 128 条 128B 小 DMA
（10× 慢 + 数值污染）。

## 3. 优化经验

### 3.1 有效的（按贡献排序）

1. **conv 替代 `__bang_matmul`**（8192: 0.79→0.39ms，2×）——但前提是结构先改对
   （见 §3.2-4 的教训：单纯换指令无收益，旧"matmul 占 0.37ms"实为结构化同步
   成本且与 MOVE 链重叠）。
2. **block_m 64→128 + 单槽化**——K/V 流量减半。
3. **深度-1 异步流水**——DMA 延迟藏进 MOVE 工作。
4. **no-max softmax**（旧主线上 1.8×）——删 max/rescale/双转置。
5. **epilogue 四件套**——除法→倒数乘、全宽 rescale、全宽 cast、逐行 store。
6. **编译器两修复**：`try_memcpy` 支持单循环形态 + 先序匹配——消灭"子视图
   标量化税"和"拷贝碎片化"两类隐性性能悬崖。
7. **batched-overlap 调度 + f32-PV**（§3.6）：MOVE 工作合并成单段 + PV 直接
   读 f32 score，8192 再从 0.417→0.315ms；fast（shift）路径全面反超 FA2。

### 3.2 证伪记录（每条都实测过）

| 尝试 | 结果 | 原因 |
|---|---|---|
| bf16 softmax 链（cast→bf16 min→bf16 exp2） | ✗ 0.186 | exp2 放大 pre-exp2 舍入；生产 logit 常达 ±15+ |
| conv 直接输出 bf16 | ✗ cnas 0117 | f32 累加器硬件硬性 |
| f32-PV conv（免 cast），**旧深度-1 调度下** | -7% | 旧调度里 cast 藏在 MOVE 尾部被部分掩盖 |
| `__bang_active_exp_less_0` 单趟 softmax | ✗ 1.7× 慢 | active 算子流 ~51 GB/s（schedule 内实测 0.539 vs 0.316 ms） |
| DMA 深度 2 / softmax 深度 3 | 持平/略差 | DMA 延迟已被掩盖；槽位分支标量税 |
| 紧凑单发 16KB store | **11×** | strided load 标量化 |
| 旧 gemm 时代：M=128 gemm、pad-K、预缩 K | ✗ | 各自实测 |
| SRAM 簇协作全家族 | ✗ | LSU 不可寻址 SRAM + DMA 慢（待 Cambricon） |

### 3.3 方法论（比结论更值钱）

1. **消融必须验证真的删掉了**：MODE 开关写在 main 里=局部变量绑定 bug，
   全部"删了没变化"都是空操作假象。判据：被删组件的 max_err 必须爆炸。
2. **差分 dump 法解 ISA**：改一个块参数重编译参考实现（Triton-MLU），
   逐字段比对 mlisa——23 参数的指令就是这样解出来的。
3. **布局制图法**：把 op 的原始输出平铺 dump 出来，宿主端搜索参考数据落点，
   直接测出真实布局（"分块布局"假象就是这么戳穿的）。
4. **改文件必须 rsync 再测**——两次踩坑（4.4× 假信号、10× 假信号都是旧代码）。
5. **单 panel 即 inf** = 快速判据：与流水线无关，直接查单 panel 路径。
6. **遍历顺序 = 匹配语义**：bottom-up 碎片化 vs pre-order 整体匹配。
7. **替换锚点必须含 docstring 首行**：同名 def 可能有前向声明（allocate.py
   被截断事故）。
8. **"看似重复的分支"可能是双缓冲语义**（v0/v1 奇偶槽）——机械折叠=引入 race。

### 3.4 剩余空间（挂外部输入）

1. **fused clamp+exp2 单算子**（Cambricon 第 7 条）：batched 调度下逐项成本
   （`probe_batchcost.py`，8192）= softmax+cast 0.181ms（44%）/ transpose+reshape
   0.075ms（18%）/ DMA 0.007ms（2%）/ conv 数学派 0.152ms（36%，折合 **113 TF**
   ——已远超 FA2 的 52 TF）。f32-PV 已兑现 cast 一半；min 一趟（~0.04ms）靠
   conv-bias shift 兑现；剩余大头是 MOVE 侧转置+reshape 的 0.075ms。
2. **SRAM 窗口**（第 3 条）：softmax 状态+filter 进 SRAM 后 MOVE 链整体消失，
   可照抄 Triton 架构（`ld.3d`/`trans.tiling` 直载 filter）。
3. cnperf 剖 cnnl kernel，实证其 51.5TF 的构成。

### 3.5 exp_less0 单趟 softmax：op 层判死，shift 语义换装重生（2026-11-02）

`__bang_active_exp_less_0`（= e^min(x,0)，2⁻²⁴ 输出下限）在 batched 调度内
复测 **1.7× 慢**（0.539 vs 0.316 ms）——active 算子流 ~51 GB/s 的实测再次
坐实，单趟方案 op 层判死。但 **conv bias shift + 纯 exp2** 的组合活了下来：
`-C` 折进 QK conv（`.add.scalar.rn`，精确），所有 score ≤ 0 后 clamp 无存在
必要 → min 一趟消失 → softmax 单趟且结构上无溢出。该 fast 路径（C=10）
8192 = 0.324 ms @53.0 TF、16384 = 1.163 @59.1 TF——**全面反超 FA2**。
契约不变：C ≥ 工作负载最大 logit。

### 3.6 batched 调度定案（2026-11-02）

1. **数值 bug 根因**：batch(m) 的 wk reshape 槽位 off-by-one（§1.4）。
   `dbg_panels.py`（kernel 逐行镜像 + 逐 panel 累加部分和 dump，参考端取
   **累计**部分和、未归一化 exp2 权重）二分定位到 panel 2——panel-0/1 正常
   而 panel≥2 全错正是"QK 复用上一 panel K filter"的指纹。
2. **成本分解**（上表 §3.4-1）：conv 数学派只占 36%。**f32-PV**（PV conv
   直接读 f32 score，`.cvt.src`）删掉整趟 cast → 8192 从 0.417→0.315ms
   （-24%）——旧调度下"-7%"的结论是调度结构造成的假象（cast 当时被部分
   掩盖，删它反而动了时序）。
3. exp2 链留在 f32 的老结论不变；cast 环节整体消失后该教训只约束 min/exp2
   两趟（不得中途降 bf16）。

完整实验矩阵、全部探针（`examples/mlu/probe_*`、`bench_*`）与 Cambricon
问题清单（7 条）见 `ai_coding/memory/tilelang_mlu590_fa_session_memory.md`。

## 3.7 后续硬化与新墙（2026-11-02 晚）

1. **shift softmax 的 ln2 bug**（已修）：`exp2(s/√d − C)` 温度差 ln2 因子，
   max_err 0.119 且逃过了 bench 的误差检查（只测 safe）。修复=log2e 同时
   折进 conv scale 与 bias。**每个 softmax 变体必须独立过数值检查。**
2. **WRAM 基址粒度**：{0, 16K, 24K} 精确；**8K 静默错数（无 fault）**；
   32K+ fault 632015。
3. **`__bang_reshape_filter` 要求 co、ci 均 %64==0** → block_n ∈ {64, 128,...}。
4. **`__bang_move` 直搬 filter**：ci=96 的 filter 布局=行主序（EXACT），
   ci=128 不是（QK 腿死）→ block_n=96 仅剩的活口在 QK filter 布局，
   待布局制图；block_n=128@dim=128 仍是数学死（双 ≥32KB filter）。
5. **KV 打包单拷贝**（主线已实现）：`[K;V]` 连续 panel 块、每 panel 一次
   DMA；尾部零填充 + `row_sum −= n_pad·pad_w` 精确修正 → seq_kv 免对齐；
   语言层 `bang._arg`/`conv._addr` 支持 BufferRegion 子视图。实测大形状
   持平——**16KB 同步拷贝是带宽主导**，修正"2×0.67µs 纯延迟"旧模型。
6. **heads/dim 扫描**：heads 至 128 线性、ratio 恒 1.18；**dim=128 是下一
   主战场（61 vs FA2 82-89 TF）**——block_n 被 ②③④ 堵死，候选=两 panel
   融合 / V 跨步直载 / Cambricon（filter 布局、基址规则、GDRAM→WRAM DMA）。
