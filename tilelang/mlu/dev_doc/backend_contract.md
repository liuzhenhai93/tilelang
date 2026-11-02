# TileLang backend contract

这篇只写补一个 **target backend** 时，与具体芯片无关的工作。硬件指令、存储和 launch 细节写在各 backend 自己的设计文档里。完整清单见 [`.agents/skills/tilelang-backend/SKILL.md`](../../../.agents/skills/tilelang-backend/SKILL.md) 和 [`tilelang/backend/README.md`](../../backend/README.md)。

## 两个名字

| 名字 | 例子 | 拥有什么 |
| --- | --- | --- |
| target backend | CUDA、ROCm、CPU、Metal | 方言、target、pass、host/device codegen、工具链回调 |
| execution backend | `tvm_ffi`、`nvrtc`、`cython`、`torch` | Build、加载、启动、cache、参数和 stream 绑定 |

包名、TVM target kind、execution 名字是三件独立的事。`tilelang/rocm` 拥有的 target kind 是 `hip`。一个 target backend 声明自己能被哪些 execution backend 消费，通常不另写一套 JIT。

一次编译只解析一次 `BackendContext`，后面的 pass 和 codegen 沿用这份结果：

```text
Language Dialect
  -> target / BackendContext
  -> PassPipeline
  -> HostCodegen + DeviceCodegen
  -> 兼容的 ExecutionBackend：Build -> Load -> Launch
```

## 开工前先写清的合同

- TVM 是否已经认识这个 target kind。不认识就先注册 kind，不用散落的字符串冒充。
- kind、属性、别名、自动探测策略。多个 manifest 共用一个 kind 时，predicate 必须互斥。
- 公共方言里先支持哪些 tile op，哪些明确拒绝。
- launch 模型：SIMT、串行，还是自定义的 program grid。
- 设备产物是源码、目标文件还是 runtime module；host 用 `c` / `llvm` 还是另写。
- `auto` 时 execution backend 的优先顺序，以及要不要 eager device compile。
- 无硬件就能做的检查，和必须上设备才能做的检查。

不要整份拷贝 CUDA。pipeline 可以参考 CPU 或 ROCm，execution 可以参考另一家。

## 要做的工作

### 1. 语言方言

目录：`tilelang/<backend>/language/`。

- 从 `tilelang.language.common` 再导出公共 API，并设置 `__tilelang_dialect__`。
- 芯片专有的同步、内存、指令放在这个包里。
- 这些 API 只构造 IR 或 annotation，不在这里打印设备源码。
- 每个新 IR 形式都要有对应的 lowering 或明确的拒绝。
- 不要把专有符号加进 `tilelang.language.common` 或默认的 `tilelang.language`。

### 2. Target 与 manifest

- `target.py`：别名、规范化、可选的自动探测。探测函数对别人的 target 返回 `None`。用户显式写的属性保留，除非文档规定了规范化。
- `backend.py`：注册唯一的 `BackendModule`。每个 owned kind 恰好一条 pipeline、一个 device codegen；至少声明一个 execution backend，声明顺序就是 `auto` 顺序。
- `__init__.py`：按 target、transform、language、intrinsic、tile op、manifest 的顺序导入，并从 `tilelang/__init__.py` 拉进来。无硬件、无 SDK 的进程也要能 import。

### 3. Pass pipeline

完整顺序写在 `tilelang/<backend>/pipeline.py`，包进名为 target kind 的 `PassPipeline`。公共入口只做 `PreLowerSemanticCheck`，然后调用 `context.lower`。

一条 pipeline 要覆盖这些边界：

1. 绑定规范化后的 target。
2. 物化 kernel launch。
3. 校验并规范化前端 IR。
4. 在 layout inference 之前做会影响 layout 的变换。
5. layout inference，再 lower tile op。
6. 内存、向量化、类型和本 target intrinsic 的合法化。
7. 分配、opaque block 和 buffer lowering。
8. 内存检查，以及本 target 支持的 reduction / sync lowering。
9. 拆 host / device。
10. storage、同步、packed API 和 launch 的收尾。

只属于这个 backend 的 Python pass 放 `tilelang/<backend>/transform/`，C++ pass 放 `src/<backend>/transform/`。可以复用另一条 pipeline，但要在 manifest 里写明，并加复用测试。

### 4. Tile op

先列出第一批 kernel 真正用到的 op，再逐个实现：

- Python 选择和注册：`tilelang/<backend>/op/`。
- C++ lowering：`src/<backend>/op/`，用 target predicate 注册。
- intrinsic 发射：`tilelang/<backend>/intrinsics/`。
- 生成源码里的 helper：`src/<backend>/codegen/` 或 `src/tl_templates/<backend>/`。

不支持的 op 在 target 解析或 lowering 时拒绝，带上 target 和缺哪项能力。不要落到别的 target 的实现上，也不要等到生成的源码编译失败才发现。

### 5. Host / device codegen

- `tilelang/<backend>/codegen.py` 提供 `DeviceCodegen`。
- 原生 backend 通常注册 `target.build.tilelang_<backend>` 和 `target.build.tilelang_<backend>_without_compile`。后者只出源码，给延迟到 Build 阶段再编译的路径用。
- 产物要带上可查看的源码、global symbol、参数类型、launch tag，以及加载和调用需要的 runtime metadata。
- host 能用 `c` / `llvm` 就复用 `STANDARD_HOST_CODEGENS`。设备侧要改 host IR 时用 `HostCodegenHook`，不要在公共 engine 里按 target kind 分支。
- C 风格 codegen 继承 `CodeGenTileLangCBase`。头文件和 helper 的输出顺序跟现有 backend 一致，不重写 `Finish()`。

### 6. Execution 兼容

在 `tilelang/<backend>/execution_backend.py` 用 `ExecutionBackendSpec` 声明兼容关系。声明某一条路径，就表示本 backend 能产出那条路径要求的源码、artifact、symbol、参数和 launch metadata。

共享的 execution backend 负责 cache、wrapper、加载、参数和 stream、launch 和生命周期。target backend 只提供源码生成、校验、编译回调和工具链。现有实现消化不了产物或 runtime API 时，才另开一个 execution backend，那是单独的一项工作。

### 7. 原生构建、测试、文档

- `src/<backend>/CMakeLists.txt` 编进本 backend 的源码。Python 包会进 wheel，C++ 源码不会自动进去。
- 无硬件测试：import、target 规范化、方言能建出 IR、pipeline 能 lower、不支持的形式有明确报错、source-only codegen 的文本断言。
- 有工具链再测编译。有设备再测加载、launch、数值和性能。
- 用户文档写清支持的 tile op、launch 模型和明确不支持的能力。

## 边界

- target 和 execution 的选择只发生在创建 `BackendContext` 时。
- 完整 pass 顺序只出现在 `tilelang/<backend>/pipeline.py`。`tilelang/engine/lower.py` 不按 target 分发。
- codegen 和工具链回调归 target backend。公共 `tilelang/backend/` 只放接口和与 target 无关的辅助。
- 真正共用、且不依赖某一家 runtime 的 C++ 辅助放 `src/backend/common/`。
- `BackendModule` 是注册清单，不是整个 backend。语言方言不在这个清单的字段里，但属于 backend。

## 完成标准

四块都有实现：方言、context/manifest、pipeline、host/device codegen，并且 execution 兼容关系已经声明和测过。复用的组件要写明是有意复用。文档列出支持项和拒绝项。只完成注册或只打出源码，不算 backend 完成。
