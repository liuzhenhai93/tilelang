/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tl/mlu/runtime/mlu_module.cc
 * \brief Load a BANG kernel and launch it with cnrtFuncTypeBlock.
 *
 * cnrtInvokeKernel takes a host function pointer, not a loose object file.
 * The source is wrapped in extern "C", linked to a shared library with cncc,
 * and the symbol is loaded with dlsym. libcnrt is opened at launch time, so
 * this file does not include cnrt.h and a build without CNToolkit still links.
 *
 * Scalar kernels use cnrtFuncTypeBlock and dim equal to the TileLang grid.
 * A cluster kernel (BANG GEMM) uses cnrtFuncTypeUnion1. Its TileLang grid
 * counts logical clusters; dim.x is that count times 4, one task per IPU.
 * taskId is not a TileLang index. Codegen recovers the cluster index as
 * taskIdX / 4.
 */

#include <dlfcn.h>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/module.h>
#include <tvm/ir/op.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "runtime/metadata.h"
#include "runtime/pack_args.h"
#include "runtime/thread_storage_scope.h"
#include "support/bytes_io.h"

namespace tvm {
namespace support {

template <>
struct Serializer<Integer> {
  static constexpr bool enabled = true;

  static void Write(Stream* strm, const Integer& data) {
    Serializer<int64_t>::Write(strm, data.IntValue());
  }

  static bool Read(Stream* strm, Integer* data) {
    int64_t value = 0;
    if (!Serializer<int64_t>::Read(strm, &value)) {
      return false;
    }
    *data = Integer(static_cast<int>(value));
    return true;
  }
};

}  // namespace support
}  // namespace tvm

namespace {

struct CnrtDim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
};

struct CnrtApi {
  void *library{nullptr};
  int (*set_device)(int){nullptr};
  int (*queue_create)(void **){nullptr};
  int (*queue_destroy)(void *){nullptr};
  int (*queue_sync)(void *){nullptr};
  int (*malloc)(void **, size_t){nullptr};
  int (*free)(void *){nullptr};
  int (*memcpy)(void *, void *, size_t, int){nullptr};
  int (*invoke)(const void *, CnrtDim3, int, void **, size_t, void *){nullptr};
};

// cnrtFuncTypeBlock. One cluster's tasks; dim.x is the task count.
constexpr int kFuncTypeBlock = 1;
constexpr int kFuncTypeUnion1 = 4;
constexpr int kMemcpyHostToDev = 0;
constexpr int kMemcpyDevToHost = 2;

bool LoadCnrt(CnrtApi *api, std::string *error) {
  const char *neuware = std::getenv("NEUWARE_HOME");
  std::string home = neuware && neuware[0] ? neuware : "/usr/local/neuware";
  std::string library = home + "/lib64/libcnrt.so";
  api->library = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (api->library == nullptr) {
    *error = dlerror();
    return false;
  }
  auto symbol = [&](const char *name) -> void * {
    return dlsym(api->library, name);
  };
  api->set_device = reinterpret_cast<int (*)(int)>(symbol("cnrtSetDevice"));
  api->queue_create = reinterpret_cast<int (*)(void **)>(symbol("cnrtQueueCreate"));
  api->queue_destroy = reinterpret_cast<int (*)(void *)>(symbol("cnrtQueueDestroy"));
  api->queue_sync = reinterpret_cast<int (*)(void *)>(symbol("cnrtQueueSync"));
  api->malloc = reinterpret_cast<int (*)(void **, size_t)>(symbol("cnrtMalloc"));
  api->free = reinterpret_cast<int (*)(void *)>(symbol("cnrtFree"));
  api->memcpy = reinterpret_cast<int (*)(void *, void *, size_t, int)>(symbol("cnrtMemcpy"));
  api->invoke = reinterpret_cast<int (*)(const void *, CnrtDim3, int, void **, size_t, void *)>(
      symbol("cnrtInvokeKernel"));
  if (!api->set_device || !api->queue_create || !api->queue_destroy || !api->queue_sync ||
      !api->malloc || !api->free || !api->memcpy || !api->invoke) {
    *error = "libcnrt is missing a launch symbol";
    return false;
  }
  return true;
}

// The official MLU-OPS kernels declare their SRAM pools at file scope.
// A function-scoped __mlu_shared__ array compiles but its address window is
// not mapped at launch (MEMORY_ILLEGAL on any access), so hoist the
// declarations next to the includes before handing the source to cncc.
bool CompileShared(const std::string &source_in, const std::string &directory, std::string *error) {
  const char *neuware = std::getenv("NEUWARE_HOME");
  std::string home = neuware && neuware[0] ? neuware : "/usr/local/neuware";
  // hoist function-scoped __mlu_shared__ declarations to file scope
  std::vector<std::string> decls;
  std::vector<std::string> keep;
  bool inserted_include = false;
  {
    std::istringstream in(source_in);
    std::string line;
    while (std::getline(in, line)) {
      size_t first = line.find_first_not_of(" \t");
      if (first != std::string::npos &&
          line.compare(first, 14, "__mlu_shared__") == 0) {
        decls.push_back(line);
      } else {
        keep.push_back(line);
      }
    }
  }
  std::ostringstream source_os;
  if (!decls.empty()) {
    for (const auto &line : keep) {
      source_os << line << "\n";
      if (!inserted_include && line.rfind("#include", 0) == 0) {
        for (const auto &d : decls) {
          source_os << d << "\n";
        }
        inserted_include = true;
      }
    }
    if (!inserted_include) {
      for (const auto &d : decls) {
        source_os << d << "\n";
      }
    }
  } else {
    source_os << source_in;
  }
  std::string source = source_os.str();
  std::string path = directory + "/kernel.mlu";
  {
    std::ofstream file(path);
    if (!file) {
      *error = "cannot write " + path;
      return false;
    }
    file << "extern \"C\" {\n" << source << "\n}\n";
  }
  // cncc's module ctor registers the fatbin (with the SRAM metadata) and
  // stores the handle in __bang_mlubin_handle; that symbol is local in the
  // object, so globalize it before linking -- the runtime dlsym's it to
  // pass the registered handle (not the stub) to cnrtInvokeKernel.
  std::ostringstream command;
  command << home << "/bin/cncc -c -fPIC -O3 -std=c++11 --bang-mlu-arch=mtp_592 -I" << home
          << "/include " << path << " -o " << directory << "/kernel.o >"
          << directory << "/cncc.log 2>&1";
  if (std::system(command.str().c_str()) != 0) {
    std::ifstream log(directory + "/cncc.log");
    std::ostringstream text;
    text << log.rdbuf();
    *error = "cncc -c failed\n" + text.str();
    return false;
  }
  std::ostringstream objcopy;
  objcopy << "objcopy --globalize-symbol=__bang_mlubin_handle " << directory
          << "/kernel.o >" << directory << "/objcopy.log 2>&1";
  if (std::system(objcopy.str().c_str()) != 0) {
    *error = "objcopy globalize failed";
    return false;
  }
  std::ostringstream link;
  link << home << "/bin/cncc -shared -fPIC -O3 -std=c++11 --bang-mlu-arch=mtp_592 "
       << directory << "/kernel.o -o " << directory << "/kernel.so -L" << home
       << "/lib64 -lcnrt >" << directory << "/cncc_link.log 2>&1";
  if (std::system(link.str().c_str()) != 0) {
    std::ifstream log(directory + "/cncc_link.log");
    std::ostringstream text;
    text << log.rdbuf();
    *error = "cncc -shared failed\n" + text.str();
    return false;
  }
  return true;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int tilelang_mlu_launch_add_one(
    const char *source, const char *symbol, int grid_x, int n) {
  if (source == nullptr || symbol == nullptr || grid_x <= 0 || n <= 0) {
    return -1;
  }
  char directory_template[] = "/tmp/tl_mlu_XXXXXX";
  char *directory = mkdtemp(directory_template);
  if (directory == nullptr) {
    return -2;
  }
  std::string error;
  if (!CompileShared(source, directory, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return -3;
  }
  std::string library_path = std::string(directory) + "/kernel.so";
  void *library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    std::fprintf(stderr, "%s\n", dlerror());
    return -4;
  }
  void *kernel = dlsym(library, symbol);
  if (kernel == nullptr) {
    std::fprintf(stderr, "missing symbol %s\n", symbol);
    dlclose(library);
    return -5;
  }

  CnrtApi cnrt;
  if (!LoadCnrt(&cnrt, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    dlclose(library);
    return -6;
  }
  if (cnrt.set_device(0) != 0) {
    dlclose(library);
    dlclose(cnrt.library);
    return -7;
  }
  void *queue = nullptr;
  if (cnrt.queue_create(&queue) != 0) {
    dlclose(library);
    dlclose(cnrt.library);
    return -8;
  }

  std::vector<float> host_in(static_cast<size_t>(n));
  std::vector<float> host_out(static_cast<size_t>(n), -1.0f);
  for (int i = 0; i < n; ++i) {
    host_in[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  void *dev_in = nullptr;
  void *dev_out = nullptr;
  size_t bytes = static_cast<size_t>(n) * sizeof(float);
  int mismatches = -9;
  if (cnrt.malloc(&dev_in, bytes) == 0 && cnrt.malloc(&dev_out, bytes) == 0 &&
      cnrt.memcpy(dev_in, host_in.data(), bytes, kMemcpyHostToDev) == 0) {
    CnrtDim3 dim{static_cast<unsigned int>(grid_x), 1u, 1u};
    void *args[] = {&dev_in, &dev_out};
    if (cnrt.invoke(kernel, dim, kFuncTypeBlock, args, 0, queue) == 0 && cnrt.queue_sync(queue) == 0 &&
        cnrt.memcpy(host_out.data(), dev_out, bytes, kMemcpyDevToHost) == 0) {
      mismatches = 0;
      for (int i = 0; i < n; ++i) {
        if (host_out[static_cast<size_t>(i)] != host_in[static_cast<size_t>(i)] + 1.0f) {
          ++mismatches;
        }
      }
    }
  }
  if (dev_in != nullptr) {
    cnrt.free(dev_in);
  }
  if (dev_out != nullptr) {
    cnrt.free(dev_out);
  }
  cnrt.queue_destroy(queue);
  dlclose(library);
  dlclose(cnrt.library);
  return mismatches;
}

namespace tvm {
namespace runtime {

namespace {

enum class KernelLaunchMode : int64_t {
  kBlock = 0,
  kUnion1 = 1,
  // Union1 with the TileLang grid counting IPU tasks directly: no x4.
  kUnion1Flat = 2,
};

class BangGemmFinder : public tirx::StmtExprVisitor {
 public:
  bool found{false};
  int max_m{0};

  void VisitExpr_(const tirx::CallNode *op) final {
    const auto *callee = op->op.as<OpNode>();
    if (callee != nullptr &&
        (callee->name == "tl.mlu_bang_gemm" || callee->name == "tl.mlu_nram_gemm")) {
      found = true;
      if (const auto *m_imm = op->args[3].as<tirx::IntImmNode>()) {
        if (m_imm->value > max_m) {
          max_m = static_cast<int>(m_imm->value);
        }
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

KernelLaunchMode InferLaunchMode(const tirx::PrimFunc &func) {
  BangGemmFinder finder;
  finder(func->body);
  if (!finder.found) {
    return KernelLaunchMode::kBlock;
  }
  // Gems are per-IPU (M <= 64 enforced by the gemm op): the grid counts
  // tasks directly; the x4 cluster row-split launch form no longer exists.
  return KernelLaunchMode::kUnion1Flat;
}

}  // namespace

class MLUModuleNode : public ffi::ModuleObj {
 public:
  MLUModuleNode(ffi::String source, ffi::Map<ffi::String, FunctionInfo> fmap, void *library,
                CnrtApi cnrt, void *queue, ffi::Map<ffi::String, Integer> launch_modes)
      : source_(std::move(source)),
        fmap_(std::move(fmap)),
        library_(library),
        cnrt_(cnrt),
        queue_(queue),
        launch_modes_(std::move(launch_modes)) {}

  ~MLUModuleNode() {
    if (queue_ != nullptr && cnrt_.queue_destroy != nullptr) {
      cnrt_.queue_destroy(queue_);
    }
    if (library_ != nullptr) {
      dlclose(library_);
    }
    if (cnrt_.library != nullptr) {
      dlclose(cnrt_.library);
    }
  }

  const char *kind() const final { return "mlu"; }

  int GetPropertyMask() const final {
    return ffi::Module::kRunnable | ffi::Module::kBinarySerializable;
  }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(source_);
    stream.Write(fmap_);
    stream.Write(launch_modes_);
    return ffi::Bytes(std::move(buffer));
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    if (format.empty() || format == "mlu") {
      return source_;
    }
    return ffi::String();
  }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final {
    auto found = fmap_.Get(name);
    if (!found.has_value()) {
      return ffi::Function();
    }
    FunctionInfo info = found.value();
    ffi::ObjectPtr<ffi::Object> self = ffi::GetObjectPtr<ffi::Object>(this);
    std::string symbol(name);
    auto launch_mode = launch_modes_.Get(name);
    TVM_FFI_CHECK(launch_mode.has_value(), RuntimeError)
        << "MLU module has no launch metadata for " << symbol;
    const int64_t mode = launch_mode.value()->value;
    bool uses_union1 = mode != static_cast<int64_t>(KernelLaunchMode::kBlock);
    bool flat_grid = mode == static_cast<int64_t>(KernelLaunchMode::kUnion1Flat);
    auto launch = [this, self, info, symbol, uses_union1, flat_grid](ffi::PackedArgs args, ffi::Any * /*rv*/,
                                                          void **kernel_args) {
      std::lock_guard<std::mutex> lock(mutex_);
      void *kernel = dlsym(library_, symbol.c_str());
      if (kernel == nullptr) {
        TVM_FFI_THROW(RuntimeError) << "MLU module is missing symbol " << symbol;
      }

      LaunchParamConfig config;
      config.Init(info->arg_types.size(), info->launch_param_tags);
      ThreadWorkLoad workload = config.Extract(args);
      for (int axis = 0; axis < 3; ++axis) {
        if (workload.grid_dim(axis) == 0 ||
            workload.grid_dim(axis) > std::numeric_limits<unsigned int>::max()) {
          TVM_FFI_THROW(RuntimeError)
              << "MLU grid extent for " << symbol << " axis " << axis
              << " is " << workload.grid_dim(axis);
        }
      }
      unsigned int grid_x = static_cast<unsigned int>(workload.grid_dim(0));
      if (uses_union1 && grid_x % 4 != 0) {
        // Union1 launches are cluster-granular: the hardware rejects a
        // task count that does not fill whole clusters (CN_ERROR 100006).
        TVM_FFI_THROW(RuntimeError)
            << "MLU Union1 launch for " << symbol << " needs grid.x ("
            << grid_x << ") to be a multiple of 4";
      }
      if (uses_union1 && !flat_grid) {
        if (grid_x > std::numeric_limits<unsigned int>::max() / 4) {
          TVM_FFI_THROW(RuntimeError)
              << "MLU cluster grid.x " << grid_x << " does not fit in four tasks per cluster";
        }
        grid_x *= 4;
      }
      CnrtDim3 dim{grid_x, static_cast<unsigned int>(workload.grid_dim(1)),
                   static_cast<unsigned int>(workload.grid_dim(2))};
      using launch_fn = int (*)(void **, unsigned, unsigned, unsigned, int, void *);
      auto tl_launch = reinterpret_cast<launch_fn>(
          dlsym(library_, "__tl_kernel_launch"));
      int status = 0;
      if (tl_launch != nullptr) {
        status = tl_launch(kernel_args, dim.x, dim.y, dim.z,
                           uses_union1 ? kFuncTypeUnion1 : kFuncTypeBlock,
                           queue_);
      } else {
        status = cnrt_.invoke(kernel, dim,
                              uses_union1 ? kFuncTypeUnion1 : kFuncTypeBlock,
                              kernel_args, 0, queue_);
      }
      if (status != 0) {
        TVM_FFI_THROW(RuntimeError)
            << "cnrtInvokeKernel failed for " << symbol << " with status " << status;
      }
      status = cnrt_.queue_sync(queue_);
      if (status != 0) {
        TVM_FFI_THROW(RuntimeError)
            << "cnrtQueueSync failed for " << symbol << " with status " << status;
      }
    };
    return PackFuncVoidAddr(launch, info->arg_types);
  }

 private:
  ffi::String source_;
  ffi::Map<ffi::String, FunctionInfo> fmap_;
  void *library_;
  CnrtApi cnrt_;
  void *queue_;
  ffi::Map<ffi::String, Integer> launch_modes_;
  std::mutex mutex_;
};

struct LoadedMLU {
  void *library;
  CnrtApi cnrt;
  void *queue;
};

LoadedMLU LoadMLUShared(const std::string &source) {
  char directory_template[] = "/tmp/tl_mlu_XXXXXX";
  char *directory = mkdtemp(directory_template);
  if (directory == nullptr) {
    TVM_FFI_THROW(RuntimeError) << "mkdtemp failed";
  }
  std::string error;
  if (!CompileShared(source, directory, &error)) {
    TVM_FFI_THROW(RuntimeError) << error;
  }
  std::string library_path = std::string(directory) + "/kernel.so";
  void *library = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    TVM_FFI_THROW(RuntimeError) << dlerror();
  }
  CnrtApi cnrt;
  if (!LoadCnrt(&cnrt, &error)) {
    dlclose(library);
    TVM_FFI_THROW(RuntimeError) << error;
  }
  if (cnrt.set_device(0) != 0) {
    dlclose(library);
    dlclose(cnrt.library);
    TVM_FFI_THROW(RuntimeError) << "cnrtSetDevice failed";
  }
  void *queue = nullptr;
  if (cnrt.queue_create(&queue) != 0) {
    dlclose(library);
    dlclose(cnrt.library);
    TVM_FFI_THROW(RuntimeError) << "cnrtQueueCreate failed";
  }
  return LoadedMLU{library, cnrt, queue};
}

ffi::Module MLUModuleCreate(IRModule mod, ffi::String source) {
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, Integer> launch_modes;
  for (auto kv : mod->functions) {
    auto func = Downcast<tirx::PrimFunc>(kv.second);
    auto calling_conv = func->GetAttr<Integer>(tvm::attr::kCallingConv);
    TVM_FFI_ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "MLU module received a non-kernel function";
    ffi::Array<DLDataType> arg_types;
    for (const auto &param : func->params) {
      DataType dtype = param.dtype();
      if (dtype.is_bool()) {
        dtype = DataType::Int(32);
      }
      arg_types.push_back(dtype);
    }
    ffi::Array<ffi::String> launch_param_tags;
    if (auto tags = func->GetAttr<ffi::Array<ffi::String>>(tirx::attr::kKernelLaunchParams)) {
      launch_param_tags = tags.value();
    }
    std::string symbol = Downcast<GlobalVar>(kv.first)->name_hint;
    if (auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      symbol = global_symbol.value();
    }
    fmap.Set(symbol, FunctionInfo(symbol, arg_types, launch_param_tags, {}));
    launch_modes.Set(symbol, Integer(static_cast<int64_t>(InferLaunchMode(func))));
  }

  LoadedMLU loaded = LoadMLUShared(source);
  return ffi::Module(ffi::make_object<MLUModuleNode>(std::move(source), std::move(fmap),
                                                     loaded.library, loaded.cnrt, loaded.queue,
                                                     std::move(launch_modes)));
}

ffi::Module MLUModuleLoadFromBytes(const ffi::Bytes &bytes) {
  support::BytesInStream stream(bytes);
  ffi::String source;
  ffi::Map<ffi::String, FunctionInfo> fmap;
  ffi::Map<ffi::String, Integer> launch_modes;
  TVM_FFI_ICHECK(stream.Read(&source));
  TVM_FFI_ICHECK(stream.Read(&fmap));
  TVM_FFI_ICHECK(stream.Read(&launch_modes));
  LoadedMLU loaded = LoadMLUShared(source);
  return ffi::Module(ffi::make_object<MLUModuleNode>(std::move(source), std::move(fmap),
                                                     loaded.library, loaded.cnrt, loaded.queue,
                                                     std::move(launch_modes)));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.mlu", MLUModuleLoadFromBytes);
}

// The MLU target uses kDLExtDev. MakePackedAPI emits __tvm_set_device for that
// type, which looks up "device_api.ext_dev". Torch owns the buffers.
class MLUDeviceAPI final : public DeviceAPI {
 public:
  void SetDevice(Device dev) final {}

  void GetAttr(Device dev, DeviceAttrKind kind, ffi::Any *rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void *AllocDataSpace(Device dev, size_t nbytes, size_t alignment, DLDataType type_hint) final {
    TVM_FFI_THROW(RuntimeError) << "MLUDeviceAPI does not allocate device memory";
    TVM_FFI_UNREACHABLE();
  }

  void FreeDataSpace(Device dev, void *ptr) final {
    TVM_FFI_THROW(RuntimeError) << "MLUDeviceAPI does not free device memory";
    TVM_FFI_UNREACHABLE();
  }

  void StreamSync(Device dev, TVMStreamHandle stream) final {}

  static MLUDeviceAPI *Global() {
    static auto *inst = new MLUDeviceAPI();
    return inst;
  }

 protected:
  void CopyDataFromTo(const void *from, size_t from_offset, void *to, size_t to_offset, size_t size,
                      Device dev_from, Device dev_to, DLDataType type_hint,
                      TVMStreamHandle stream) final {
    TVM_FFI_THROW(RuntimeError) << "MLUDeviceAPI does not copy device memory";
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed("device_api.ext_dev", [](ffi::PackedArgs args, ffi::Any *rv) {
    *rv = static_cast<void *>(MLUDeviceAPI::Global());
  });
}

}  // namespace runtime
}  // namespace tvm
