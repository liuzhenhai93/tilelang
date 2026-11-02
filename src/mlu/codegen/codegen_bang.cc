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
 * \file tl/mlu/codegen/codegen_bang.cc
 * \brief Scalar BANG C for an MLU device kernel.
 *
 * A cluster kernel's blockIdx.x is the logical cluster index, printed as
 * (taskIdX / 4). The runtime launches four tasks per cluster. blockIdx.y/z
 * print as taskIdY/Z. A scalar Block kernel prints blockIdx.x as taskIdX.
 * threadIdx is rejected. coreId is taskIdX % 4 inside the cluster helper.
 * The compiled entry calls
 * tilelang_callback_mlu_compile and checks that cncc returned bytes, then
 * returns a runnable module. The source-only entry still returns C source.
 */

#include <cmath>
#include <string>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/stmt_functor.h>

#include "backend/common/codegen/codegen_c_base.h"
#include "support/check.h"

namespace tvm {
namespace runtime {
ffi::Module MLUModuleCreate(IRModule mod, ffi::String source);
}  // namespace runtime

namespace codegen {

class CodeGenBANG : public CodeGenTileLangCBase {
public:
  void PrintFuncPrefix(std::ostream &os) final { os << "__mlu_global__ "; }

  void PrintType(DataType t, std::ostream &os) final {
    if (t.is_bfloat16()) {
      os << "bfloat16_t";
      return;
    }
    CodeGenC::PrintType(t, os);
  }

  std::string CastFromTo(std::string value, DataType from, DataType target) final {
    if (from.is_bfloat16() && target == DataType::Float(32)) {
      return "__bfloat162float(" + value + ")";
    }
    if (target.is_bfloat16() && from == DataType::Float(32)) {
      return "__float2bfloat16(" + value + ")";
    }
    return CodeGenC::CastFromTo(std::move(value), from, target);
  }

  void VisitExpr_(const tirx::FloatImmNode *op, std::ostream &os) final {
    if (op->dtype.is_float() && op->dtype.bits() == 32 && std::isinf(op->value)) {
      os << (op->value < 0 ? "(-1.0f/0.0f)" : "(1.0f/0.0f)");
      return;
    }
    CodeGenC::VisitExpr_(op, os);
  }

  void VisitExpr_(const tirx::MaxNode *op, std::ostream &os) final {
    os << "fmaxf(";
    PrintExpr(op->a, os);
    os << ", ";
    PrintExpr(op->b, os);
    os << ")";
  }

  void VisitExpr_(const tirx::CallNode *op, std::ostream &os) final {
    const auto *callee = op->op.as<OpNode>();
    if (callee != nullptr &&
        (callee->name == "tir.call_extern" || callee->name == "tirx.call_extern") &&
        op->args.size() >= 1) {
      if (auto func = op->args[0].as<tirx::StringImm>()) {
        const std::string name = func.value()->value;
        if (name == "taskIdX") {
          os << "(taskIdX)";
          return;
        }
        if (name == "taskDim") {
          os << "taskDim";
          return;
        }
        if (name == "coreId") {
          os << "coreId";
          return;
        }
        if (name == "clusterId") {
          os << "clusterId";
          return;
        }
        if (name == "GDRAM2NRAM" || name == "NRAM2GDRAM" || name == "NRAM2NRAM" || name == "GDRAM2SRAM" || name == "SRAM2NRAM" || name == "SRAM2GDRAM" || name == "NRAM2WRAM") {
          os << name;
          return;
        }
        if (name.rfind("__tl_mlu_ew_", 0) == 0) {
          uses_elementwise_ = true;
        }
        if (name.rfind("__tl_mlu_reduce", 0) == 0) {
          if (name.find("sum") != std::string::npos) {
            uses_reduce_sum_ = true;
          } else {
            uses_reduce_max_ = true;
          }
        }
        if (name == "__bang_transpose") {
          uses_transpose_ = true;
        }
      }
    }
    if (callee != nullptr && callee->name == "tl.mlu_nram_gemm") {
      ICHECK(op->args.size() == 10U || op->args.size() == 13U);
      // Ten args: the task form with a caller-allocated WRAM scratch; the
      // M/N/K/clear/trans_b indices shift by one after the scratch pointer.
      // Thirteen args: the full form (wram + a_panel + b_nk + c_part).
      const int m_off = op->args.size() == 10U ? 1 : 4;
      const bool bf16 = PointerElementIsBf16(op->args[1]);
      const auto *n_imm = op->args[4 + m_off].as<tirx::IntImmNode>();
      const auto *k_imm = op->args[5 + m_off].as<tirx::IntImmNode>();
      if (n_imm != nullptr && k_imm != nullptr) {
        const int64_t nk =
            static_cast<int64_t>(n_imm->value) * k_imm->value;
        int &slot = bf16 ? nram_gemm_bf16_maxnk_ : nram_gemm_maxnk_;
        if (nk > slot && nk <= (1 << 16)) {
          slot = static_cast<int>(nk);
        }
      }
      const auto *m_imm = op->args[3 + m_off].as<tirx::IntImmNode>();
      if (bf16) {
        uses_nram_gemm_bf16_ = true;
        const auto *clear_imm = op->args[6 + m_off].as<tirx::IntImmNode>();
        const auto *trans_imm = op->args[7 + m_off].as<tirx::IntImmNode>();
        if (n_imm == nullptr || k_imm == nullptr || clear_imm == nullptr ||
            trans_imm == nullptr || m_imm == nullptr || trans_imm->value == 0 ||
            static_cast<int64_t>(n_imm->value) * k_imm->value > (1 << 14) ||
            (!clear_imm->value &&
             m_imm->value * n_imm->value > 8192)) {
          nram_gemm_bf16_lean_ok_ = false;
        }
      } else {
        uses_nram_gemm_ = true;
      }
      os << (bf16 ? "__tl_mlu_nram_gemm_bf16_task(" : "__tl_mlu_nram_gemm_task(");
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i != 0) {
          os << ", ";
        }
        os << PrintExpr(op->args[i]);
      }
      os << ")";
      return;
    }
    if (callee != nullptr && callee->name == "tl.mlu_bang_gemm") {
      ICHECK_EQ(op->args.size(), 7U);
      uses_bang_gemm_ = true;
      os << "__tl_mlu_bang_gemm(";
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i != 0) {
          os << ", ";
        }
        os << PrintExpr(op->args[i]);
      }
      os << ")";
      return;
    }
    CodeGenTileLangCBase::VisitExpr_(op, os);
  }

  void SetClusterProgram(bool cluster_program) { cluster_program_ = cluster_program; }

  void PrintStorageScope(const std::string &scope, std::ostream &os) final {
    if (scope == "nram" || scope == "local" || scope == "local.var" || scope == "local.fragment") {
      os << "__nram__ ";
      return;
    }
    if (scope == "wram" || scope == "local.wram") {
      os << "__wram__ ";
      return;
    }
    if (scope == "shared" || scope == "shared.dyn") {
      os << "__mlu_shared__ ";
      return;
    }
    ICHECK_EQ(scope, "global") << "Unsupported MLU storage scope: " << scope;
  }

  void BindThreadIndex(const IterVar &iv) final {
    ICHECK(!var_idmap_.count(iv->var.get()));
    const std::string tag = iv->thread_tag;
    std::string builtin;
    if (tag == "blockIdx.x") {
      // Union1 packs four IPU tasks into one logical cluster. taskId stays
      // in the launch and in this expression; the kernel variable is bx.
      builtin = cluster_program_ ? "(taskIdX / 4)" : "taskIdX";
    } else if (tag == "blockIdx.y") {
      builtin = "taskIdY";
    } else if (tag == "blockIdx.z") {
      builtin = "taskIdZ";
    }
    ICHECK(!builtin.empty())
        << "MLU codegen has no thread `" << tag
        << "`. T.Kernel is a grid of programs, not SIMT threads.";
    var_idmap_[iv->var.get()] =
        CastFromTo(builtin, DataType::UInt(32), iv->var.dtype());
  }

protected:
  void EmitPreamble_(std::ostream &os) const final {
    os << "#include <cstdint>\n";
    // IPU-core-only barrier (the raw `sync` instruction): does not include
    // the MPU context, unlike __sync_cluster. Emitted unconditionally so
    // kernels without other BANG features can still use it.
    os << "#include <bang_device_types.h>\n";
    os << "__mlu_func__ void __tl_mlu_sync_ipu(void) {"
          " __asm__ volatile(\"sync;\\n\\t\"); }\n";
    // 1x1-convolution matmul on the TFU: C[m, co] = A[m, ci] @ W[co, ci].
    // Operand encoding decoded from the working Triton-MLU dump
    // (fa_kernel.mlisa, 2026-11-01): the positional params after the three
    // buffers are (ci, m/4, 1, 1, 1, 1, 1, co) and .kstride is the filter
    // row stride in BYTES (= ci*2 for bf16) as an immediate. Specialized to
    // the FlashAttention geometry: QK has ci=64/co=128 (kstride 128), PV
    // has ci=128/co=64 (kstride 256). .mul.scalar folds a scale into the
    // QK; .add.partial accumulates into C for the PV.
    os << "__mlu_func__ void __tl_mlu_conv_qk(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, int m) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 64, %3, 1, 1, 1, 1, 1, 128, .mstride(1), .kstride(128),"
          " .indilation(1, 1), .mul.scalar.rn(%4);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(m), \"r\"(scale));"
          " }\n";
    // Chunked variant: Triton always splits M into four conv calls
    // (M=128 -> 4x32 rows, M=64 -> 4x16 rows); param2 is the per-call
    // row count. Loop in C with explicit pointer advance.
    os << "__mlu_func__ void __tl_mlu_conv_qk4(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, int m) {"
          " int chunk = m >> 2;"
          " for (int t = 0; t < 4; ++t) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 64, %3, 1, 1, 1, 1, 1, 128, .mstride(1), .kstride(128),"
          " .indilation(1, 1), .mul.scalar.rn(%4);\""
          " : : \"r\"(c + t * chunk * 128), \"r\"(a + t * chunk * 64),"
          " \"r\"(w), \"r\"(chunk), \"r\"(scale));"
          " }"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_pv(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, int m) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, 64, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]), .cvt.src.rn.f32();\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(m));"
          " }\n";
    // PV variants: (a) single call without .cvt.src (our input is already
    // bf16, Triton's was f32 -- cvt.src may assume an f32 source);
    // (b) Triton's actual pattern: four chunked calls, .add.partial
    // accumulating chunk by chunk.
    os << "__mlu_func__ void __tl_mlu_conv_pv_ncv(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, int m) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, 64, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(m));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_pv4(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, int m) {"
          " int chunk = m >> 2;"
          " for (int t = 0; t < 4; ++t) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, 64, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]), .cvt.src.rn.f32();\""
          " : : \"r\"(c + t * chunk * 64), \"r\"(a + t * chunk * 128),"
          " \"r\"(w), \"r\"(chunk));"
          " }"
          " }\n";
    // Direct SRAM -> WRAM loads in the conv-filter layout, operands and
    // parameters verbatim from the working Triton-MLU dump (FA geometry:
    // K filter [co=128][ci=64]; transposed V filter [co=64][ci=128]).
    // The tiling parameters derive from %__wram_size__ (>>4 and <<3 of the
    // >>4 value); read it into an allocated gpr instead of naming physical
    // registers, so the compiler's allocator stays consistent.
    os << "__mlu_func__ void __tl_mlu_ld3d_k(__wram__ bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " int ws; __asm__ volatile(\"mv.gpr.sreg %0, %%__wram_size__;\" : \"=r\"(ws));"
          " long r124 = (long)(ws >> 4);"
          " __asm__ volatile(\"ld.3d.w16.wram.sram [%0], [%1],"
          " 512, %2, 15, 512, 1, 512, 15, 8192, 1;\""
          " : : \"r\"(dst), \"r\"(src), \"r\"(r124));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_trans_tiling_v(__wram__ bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " int ws; __asm__ volatile(\"mv.gpr.sreg %0, %%__wram_size__;\" : \"=r\"(ws));"
          " int r27 = ws >> 4; int r258 = r27 << 3;"
          " __asm__ volatile(\"trans.tiling.wram.sram.b16 [%0], [%1],"
          " 64, 32, 128, 1, 64, 4, 4096, 2, 64, 1, 128, 64, 4, 256, 8,"
          " %2, 4, 64, 2, %3, 1, 1024;\""
          " : : \"r\"(dst), \"r\"(src), \"r\"(r27), \"r\"(r258));"
          " }\n";
    // Debug aid: bit-cast the %__wram_size__ sreg into a float buffer.
    os << "__mlu_func__ void __tl_mlu_read_wram_size(float* out) {"
          " int ws; __asm__ volatile(\"mv.gpr.sreg %0, %%__wram_size__;\" : \"=r\"(ws));"
          " *((int*)out) = ws;"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_read_nram_size(float* out) {"
          " int ns; __asm__ volatile(\"mv.gpr.sreg %0, %%__nram_size__;\" : \"=r\"(ns));"
          " *((int*)out) = ns;"
          " }\n";
    // LSU load SRAM -> NRAM (Triton's staging read path). Decides whether
    // the task's LSU can address SRAM at all in the dlsym+invoke mode.
    os << "__mlu_func__ void __tl_mlu_ld_nram_sram(bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " __asm__ volatile(\"ld.stride.nram.sram [%0], [%1],"
          " 16384, 16384, 16384, 0;\" : : \"r\"(dst), \"r\"(src));"
          " }\n";
    // Row-sum pool reduction over the last dim: params (m/4, w, 1, w,
    // 1, 1, 1, 1) per the Triton-MLU dump; f32 in, f32 [m] out.
    os << "__mlu_func__ void __tl_mlu_pool_sum(float* dst, const float* src,"
          " int m4, int w) {"
          " __asm__ volatile(\"pool.sum.mstride.nram.f32 [%0], [%1],"
          " %2, %3, 1, %3, 1, 1, 1, 1;\" : : \"r\"(dst), \"r\"(src),"
          " \"r\"(m4), \"r\"(w));"
          " }\n";
    // General-register-parameter variants: rows and co as runtime gprs
    // (ci stays literal because .kstride is an immediate tied to it).
    os << "__mlu_func__ void __tl_mlu_conv_gk64(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, int rows, int co) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 64, %3, 1, 1, 1, 1, 1, %4, .mstride(1), .kstride(128),"
          " .indilation(1, 1), .mul.scalar.rn(%5);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(rows), \"r\"(co), \"r\"(scale));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_gk128(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, int rows, int co) {"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, %4, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(rows), \"r\"(co));"
          " }\n";
    // PV variants reading an f32 probability matrix directly (skips the
    // bf16 cast pass): with and without the .cvt.src source-convert
    // modifier Triton pairs with its f32 p operand.
    os << "__mlu_func__ void __tl_mlu_conv_gk128f32(float* c, const float* a,"
          " const __wram__ bfloat16_t* w, int rows, int co, int cvt) {"
          " if (cvt)"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, %4, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]), .cvt.src.rn.f32();\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(rows), \"r\"(co));"
          " else"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " 128, %3, 1, 1, 1, 1, 1, %4, .mstride(1), .kstride(256),"
          " .indilation(1, 1), .add.partial.rn([%0]);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(rows), \"r\"(co));"
          " }\n";
    // QK conv writing a bf16 score directly (bf16 softmax chain saves
    // half the MOVE traffic of the f32 chain). Same encoding, bf16 dst.
    os << "__mlu_func__ void __tl_mlu_conv_qk_bf16o(bfloat16_t* c,"
          " const bfloat16_t* a, const __wram__ bfloat16_t* w,"
          " float scale, int rows, int co) {"
          " __asm__ volatile(\"conv.nram.rn.bf16.bf16.bf16 [%0], [%1], [%2],"
          " 64, %3, 1, 1, 1, 1, 1, %4, .mstride(1), .kstride(128),"
          " .indilation(1, 1), .mul.scalar.rn(%5);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(rows), \"r\"(co), \"r\"(scale));"
          " }\n";
    // Fully generic conv: ci/rows/co/kstride all runtime registers, the
    // modifier accepts a register operand (verified on-device,
    // probe_convgen.py: ci=64/128 both exact). One helper covers every ci;
    // accumulate selects .add.partial, cvt selects the f32-input
    // .cvt.src form (the bf16 operand pointer is reinterpreted as float*
    // locally -- the two views share the address).
    os << "__mlu_func__ void __tl_mlu_conv_gen(float* c, const void* a,"
          " const __wram__ bfloat16_t* w, float scale, int ci, int rows,"
          " int co, int accumulate, int cvt) {"
          " int ks = ci << 1;"
          " const bfloat16_t* ab = (const bfloat16_t*)a;"
          " const float* af = (const float*)a;"
          " if (accumulate && cvt)"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .add.partial.rn([%0]), .cvt.src.rn.f32();\""
          " : : \"r\"(c), \"r\"(ab), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks));"
          " else if (accumulate)"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .add.partial.rn([%0]);\""
          " : : \"r\"(c), \"r\"(ab), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks));"
          " else if (cvt)"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .cvt.src.rn.f32();\""
          " : : \"r\"(c), \"r\"(af), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale));"
          " else"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7);\""
          " : : \"r\"(c), \"r\"(ab), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale));"
          " }\n";
    // Bias-fusion spelling probes: four candidate modifier spellings for
    // adding a scalar bias inside the conv instruction. If any assembles
    // and computes, the no-max softmax collapses to one exp_less0 pass
    // (the QK conv then emits (q@k^T)*scale - C directly).
    os << "__mlu_func__ void __tl_mlu_conv_bias_as(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, float bias, int ci,"
          " int rows, int co) {"
          " int ks = ci << 1;"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .add.scalar.rn(%8);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_bias_a(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, float bias, int ci,"
          " int rows, int co) {"
          " int ks = ci << 1;"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .add.scalar(%8);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_bias_br(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, float bias, int ci,"
          " int rows, int co) {"
          " int ks = ci << 1;"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .bias.rn(%8);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_conv_bias_b(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, float bias, int ci,"
          " int rows, int co) {"
          " int ks = ci << 1;"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .bias(%8);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " }\n";
    // QK conv with a fused BIAS: (q@k^T)*scale + bias in one instruction
    // (verified: the .add.scalar.rn modifier assembles and computes
    // exactly, probe_convbias.py). This is the enabler for the one-pass
    // exp_less0 softmax: bias = -shift puts the exp argument in range.
    os << "__mlu_func__ void __tl_mlu_conv_gen_b(float* c, const bfloat16_t* a,"
          " const __wram__ bfloat16_t* w, float scale, float bias, int ci,"
          " int rows, int co, int cvt) {"
          " int ks = ci << 1;"
          " const float* af = (const float*)a;"
          " if (cvt)"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .add.scalar.rn(%8),"
          " .cvt.src.rn.f32();\""
          " : : \"r\"(c), \"r\"(af), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " else"
          " __asm__ volatile(\"conv.nram.rn.f32.bf16.bf16 [%0], [%1], [%2],"
          " %3, %4, 1, 1, 1, 1, 1, %5, .mstride(1), .kstride(%6),"
          " .indilation(1, 1), .mul.scalar.rn(%7), .add.scalar.rn(%8);\""
          " : : \"r\"(c), \"r\"(a), \"r\"(w), \"r\"(ci), \"r\"(rows), \"r\"(co),"
          " \"r\"(ks), \"r\"(scale), \"r\"(bias));"
          " }\n";
    // Raw pio-tagged barrier: the SRAM-era finding says the pio/cio tags
    // drain DMA queues. Probe whether it fences an async GDRAM->NRAM load.
    os << "__mlu_func__ void __tl_mlu_bar_pio_4(void) {"
          " __asm__ volatile(\"barrier.sync.local.pio.pmv.cio.cmv 0, 4;\\n\\t\"); }"
          "\n";
    os << "__mlu_func__ void __tl_mlu_bar_pio_5(void) {"
          " __asm__ volatile(\"barrier.sync.local.pio.pmv.cio.cmv 0, 5;\\n\\t\"); }"
          "\n";
    // IO-PIPE load instructions (the guide's ld/ld.async family -- NOT the
    // DMA engine). Triton-MLU stages tensors with these; unlike the
    // __memcpy_async DMA ops they are fenced by the sync instructions.
    // Forms copied from the Triton fa_kernel.mlisa dump.
    os << "__mlu_func__ void __tl_mlu_ld_nram_gdram(bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " __asm__ volatile(\"ld.stride.nram.gdram [%0], [%1],"
          " 16384, 16384, 16384, 0;\" : : \"r\"(dst), \"r\"(src));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_ld_async_nram_gdram(bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " __asm__ volatile(\"ld.async.stride.nram.gdram [%0], [%1],"
          " 16384, 16384, 16384, 0;\" : : \"r\"(dst), \"r\"(src));"
          " }\n";
    os << "__mlu_func__ void __tl_mlu_ld_sram_gdram(bfloat16_t* dst,"
          " const bfloat16_t* src) {"
          " __asm__ volatile(\"ld.nopred.stride.sram.gdram [%0], [%1],"
          " 16384, 16384, 16384, 0;\" : : \"r\"(dst), \"r\"(src));"
          " }\n";
    // Un-tiler for the conv engine's output -> row-major
    // f32 [64][128]; NRAM -> NRAM only (no SRAM addressing). Parameters
    // verbatim from the Triton-MLU dump's BM=64/BN=128/DIM=64 kernel
    // (the same geometry as our FlashAttention).
    os << "__mlu_func__ void __tl_mlu_untilt_qk(float* dst, const float* src) {"
          " __asm__ volatile(\"trans.tiling.nram.b32 [%0], [%1],"
          " 64, 16, 64, 1, 4096, 4, 1024, 1, 64, 1, 4096,"
          " 64, 16, 256, 1, 4096, 4, 64, 1, 4096, 1, 4096;\""
          " : : \"r\"(dst), \"r\"(src));"
          " }\n";
    if (!uses_bang_gemm_ && !uses_nram_gemm_ && !uses_nram_gemm_bf16_ &&
        !uses_elementwise_ && !uses_reduce_max_ && !uses_reduce_sum_ && !uses_transpose_) {
      return;
    }
    // mlu.h is the umbrella BANG header (types + device functions + math
    // extensions such as __bang_expand_*); cncc only auto-includes the base
    // device functions.
    os << "#include <mlu.h>\n";
    if (uses_elementwise_) {
      os << R"(

// Contiguous NRAM elementwise helpers. Callers guarantee count is a positive
// multiple of 64, the BANG NFU granularity on MLU590. The generated source is
// wrapped in extern "C", so dtype-specific names are used instead of templates.
#define TL_MLU_EW_DEFINE(T, suffix)                                                   \
__mlu_func__ void __tl_mlu_ew_##suffix##_fill(T* dst, T value, int32_t count) {      \
  __bang_write_value(dst, count, value);                                               \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_add(                                         \
    T* dst, const T* lhs, const T* rhs, int32_t count) {                              \
  __bang_add(dst, lhs, rhs, count);                                                    \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_sub(                                         \
    T* dst, const T* lhs, const T* rhs, int32_t count) {                              \
  __bang_sub(dst, lhs, rhs, count);                                                    \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_mul(                                         \
    T* dst, const T* lhs, const T* rhs, int32_t count) {                              \
  __bang_mul(dst, lhs, rhs, count);                                                    \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_max(                                         \
    T* dst, const T* lhs, const T* rhs, int32_t count) {                              \
  __bang_maxequal(dst, lhs, rhs, count);                                               \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_mul_scalar(                                  \
    T* dst, const T* src, T scalar, int32_t count) {                                  \
  __bang_mul_scalar(dst, src, scalar, count);                                         \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_add_scalar(                                  \
    T* dst, const T* src, T scalar, int32_t count) {                                  \
  __bang_add_scalar(dst, src, scalar, count);                                         \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_sub_scalar(                                  \
    T* dst, const T* src, T scalar, int32_t count) {                                  \
  __bang_sub_scalar(dst, src, scalar, count);                                         \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_max_scalar(                                  \
    T* dst, T* src, T scalar, int32_t count) {                                  \
  __bang_maximum_scalar(dst, src, scalar, count);                                     \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_min_scalar(                                  \
    T* dst, T* src, T scalar, int32_t count) {                                  \
  __bang_minimum_scalar(dst, src, scalar, count);                                     \
}                                                                                      \
__mlu_func__ void __tl_mlu_ew_##suffix##_exp2(                                        \
    T* dst, const T* src, int32_t count) {                                            \
  __bang_pow2(dst, src, count);                                                        \
}
TL_MLU_EW_DEFINE(float, f32)
TL_MLU_EW_DEFINE(bfloat16_t, bf16)
#undef TL_MLU_EW_DEFINE
)";;
    }
    if (uses_reduce_max_) {
      os << R"(

// One contiguous float row. n is a positive multiple of 64. The row loop stays
// outside these helpers so the cluster pass can keep each IPU on its own rows.
__mlu_func__ void __tl_mlu_reduce_max(float* dst, const float* src, int32_t n, int32_t clear) {
  __nram__ float partial[64];
  __bang_maxpool(partial, src, 1, 1, (uint32_t)n, 1, (uint32_t)n, 1, 1);
  *dst = clear ? partial[0] : fmaxf(*dst, partial[0]);
}
// One 2D float panel [rows, cols] with cols a positive multiple of 64. One
// pool instruction reduces every row at once (kernel spans the full width,
// stride = kernel, so windows do not overlap). rows is capped by the GEMM M
// limit of 256. With clear=0 the scalar loop combines the pooled row values
// with the previous dst contents.
__mlu_func__ void __tl_mlu_reduce2d_max(float* dst, const float* src, int32_t rows, int32_t cols, int32_t clear) {
  __nram__ float partial[256];
  __bang_maxpool(partial, src, 1, (uint32_t)rows, (uint32_t)cols, 1, (uint32_t)cols);
  if (clear) {
    __bang_move(dst, partial, rows * sizeof(float));
  } else {
    for (int32_t i = 0; i < rows; ++i) {
      dst[i] = fmaxf(dst[i], partial[i]);
    }
  }
}
__mlu_func__ void __tl_mlu_reduce2d_sum(float* dst, const float* src, int32_t rows, int32_t cols, int32_t clear) {
  for (int32_t r = 0; r < rows; ++r) {
    float partial = __bang_sum(src + r * cols, (uint32_t)cols);
    dst[r] = clear ? partial : (dst[r] + partial);
  }
}
// Axis-0 pool reduce on a [rows, cols] panel: dst[j] = reduce over src[:, j].
// The kernel spans the full height with stride = height, so each output
// column collapses to one value. cols is capped by block_m (256).
// Axis-0 reduce on a [rows, cols] panel via a halving tree of PLAIN
// streaming ops (__bang_maxequal / __bang_add): the two halves of the first
// axis are contiguous blocks, so log2(rows) fast passes replace the slow
// reduce engine entirely. Requires rows to be a power of two and
// rows*cols <= 16384 for the ping-pong scratch; otherwise fall back to the
// per-row slow-engine calls.
__mlu_func__ void __tl_mlu_reduce2da_max(float* dst, const float* src, int32_t rows, int32_t cols, int32_t clear) {
  if (((rows & (rows - 1)) != 0) || rows * cols > 8192 || rows < 2) {
    for (int32_t c = 0; c < cols; ++c) {
      float acc = src[c];
      for (int32_t r = 1; r < rows; ++r) {
        acc = fmaxf(acc, src[r * cols + c]);
      }
      dst[c] = clear ? acc : fmaxf(dst[c], acc);
    }
    return;
  }
  __nram__ float bufA[4096];
  __nram__ float bufB[2048];
  int32_t h = rows;
  __bang_maxequal(bufA, src, src + h / 2 * cols, h / 2 * cols);
  h /= 2;
  const float* in = bufA;
  float* out = bufB;
  while (h > 1) {
    __bang_maxequal(out, in, in + h / 2 * cols, h / 2 * cols);
    h /= 2;
    const float* t = in;
    in = out;
    out = (float*)t;
  }
  if (clear) {
    __bang_move(dst, in, cols * sizeof(float));
  } else {
    for (int32_t i = 0; i < cols; ++i) {
      dst[i] = fmaxf(dst[i], in[i]);
    }
  }
}

)";;
    }
    if (uses_reduce_sum_) {
      os << R"(
// Pairwise-halving sum tree over a whole [rows, cols] panel: log2(cols)
// __bang_reduce_sum calls replace one call per row. The scratch ping-pongs
// between the first two levels' sizes (rows*cols/2 and rows*cols/4 floats).
__mlu_func__ void __tl_mlu_reduce2d_tree_sum(float* dst, const float* src, int32_t rows, int32_t cols, int32_t clear) {
  __nram__ float bufA[4096];
  __nram__ float bufB[2048];
  __bang_reduce_sum(bufA, src, (uint32_t)(rows * cols));
  int32_t width = cols / 2;
  const float* in = bufA;
  float* out = bufB;
  while (width > 1) {
    __bang_reduce_sum(out, in, (uint32_t)(rows * width));
    width /= 2;
    const float* t = in;
    in = out;
    out = (float*)t;
  }
  if (clear) {
    __bang_move(dst, in, rows * sizeof(float));
  } else {
    for (int32_t i = 0; i < rows; ++i) {
      dst[i] = dst[i] + in[i];
    }
  }
}
__mlu_func__ void __tl_mlu_reduce_sum(float* dst, const float* src, int32_t n, int32_t clear) {
  float partial = __bang_sum(src, (uint32_t)n);
  *dst = clear ? partial : (*dst + partial);
}
__mlu_func__ void __tl_mlu_reduce2da_sum(float* dst, const float* src, int32_t rows, int32_t cols, int32_t clear) {
  if (((rows & (rows - 1)) != 0) || rows * cols > 8192 || rows < 2) {
    for (int32_t c = 0; c < cols; ++c) {
      float acc = src[c];
      for (int32_t r = 1; r < rows; ++r) {
        acc = acc + src[r * cols + c];
      }
      dst[c] = clear ? acc : (dst[c] + acc);
    }
    return;
  }
  __nram__ float bufA[4096];
  __nram__ float bufB[2048];
  int32_t h = rows;
  __bang_add(bufA, src, src + h / 2 * cols, h / 2 * cols);
  h /= 2;
  const float* in = bufA;
  float* out = bufB;
  while (h > 1) {
    __bang_add(out, in, in + h / 2 * cols, h / 2 * cols);
    h /= 2;
    const float* t = in;
    in = out;
    out = (float*)t;
  }
  if (clear) {
    __bang_move(dst, in, cols * sizeof(float));
  } else {
    for (int32_t i = 0; i < cols; ++i) {
      dst[i] = dst[i] + in[i];
    }
  }
}
)";;
    }
    if (uses_bang_gemm_) {
    os << R"(

__mlu_func__ void __tl_mlu_bang_gemm(float* C, const float* A,
                                      const float* B, int32_t M, int32_t N,
                                      int32_t K, bool clear_accum) {
  constexpr int32_t kTile = 64;
  __nram__ float a_nram[64 * 64];
  __nram__ float b_kn[64 * 64];
  __nram__ float b_nk[64 * 64];
  __nram__ float b_filter[64 * 64];
  __nram__ float c_tile[64 * 64];
  // One N=64 panel; every task owns its whole M-row tile (M <= 64) and
  // streams its own B tiles: no MPU/SRAM staging (SRAM is unavailable
  // under the flat task launch).
  __nram__ float c_panel[64 * 64];
  __wram__ float b_wram[64 * 64];
  const int32_t rows = M;
  const int32_t row_begin = 0;
  for (int32_t n0 = 0; n0 < N; n0 += kTile) {
    if (clear_accum) {
      __bang_write_value(c_panel, rows * kTile, 0.0f);
    } else {
      for (int32_t row = 0; row < rows; ++row) {
        __memcpy(c_panel + row * kTile, C + (row_begin + row) * N + n0,
                 kTile * sizeof(float), GDRAM2NRAM);
      }
    }
    for (int32_t k0 = 0; k0 < K; k0 += kTile) {
      for (int32_t k = 0; k < kTile; ++k) {
        __memcpy(b_kn + k * kTile, B + (k0 + k) * N + n0,
                 kTile * sizeof(float), GDRAM2NRAM);
      }
      // B tile is [K, N]. __bang_matmul wants the filter as [N, K] in WRAM.
      __bang_transpose(b_nk, b_kn, kTile, kTile);
      __bang_reshape_filter(b_filter, b_nk, kTile, 1, 1, kTile);
      __memcpy(b_wram, b_filter, kTile * kTile * sizeof(float), NRAM2WRAM);
      for (int32_t row = 0; row < rows; ++row) {
        __memcpy(a_nram + row * kTile, A + (row_begin + row) * K + k0,
                 kTile * sizeof(float), GDRAM2NRAM);
      }
      __bang_matmul(c_tile, a_nram, b_wram, rows, kTile, kTile);
        __bang_add(c_panel, c_panel, c_tile, rows * kTile);
    }
    for (int32_t row = 0; row < rows; ++row) {
      __memcpy(C + (row_begin + row) * N + n0, c_panel + row * kTile,
               kTile * sizeof(float), NRAM2GDRAM);
    }
  }
}
)";
    }
    if (uses_nram_gemm_bf16_ && nram_gemm_bf16_lean_ok_) {
      os << ExpandTag(R"(

// bfloat16 A and B, every call site a trans_b single shot: the staging
// panels (a_panel/b_nk) are never touched, so the lean helper only carries
// the WRAM filter and the accumulate scratch.
__mlu_func__ void __tl_mlu_nram_gemm_bf16_impl(float* C, const bfloat16_t* A,
                                               const bfloat16_t* B, int32_t M,
                                               int32_t N, int32_t K,
                                               bool clear_accum, bool trans_b,
                                               int32_t row_begin, int32_t rows,
                                               __wram__ bfloat16_t* b_wram) {
  if (__is_mpu()) {
    return;
  }
  if (clear_accum) {
    // Zero then accumulate: the stream-muladd instruction is ~4x the
    // per-FLOP rate of plain __bang_matmul on mtp_592, so even a clear
    // gemm runs through it (C is pre-zeroed, so the add is a no-op).
    __bang_write_value(C + row_begin * N, rows * N, 0.0f);
    __bang_reshape_filter(b_wram, B, N, 1, 1, K, NRAM2WRAM);
    __sync_move();
    __mlvm_stream_matmul_muladd_scal_part_f32_bf16_bf16_f32(
        C + row_begin * N, A + row_begin * K, b_wram, 1.0f,
        C + row_begin * N, rows, K, N);
  } else {
    // Fused accumulate: C += A@B in one matmul post-op, so no c_part
    // scratch (32 KB NRAM) and no separate 8192-wide add.
    __bang_reshape_filter(b_wram, B, N, 1, 1, K, NRAM2WRAM);
    __sync_move();
    __mlvm_stream_matmul_muladd_scal_part_f32_bf16_bf16_f32(
        C + row_begin * N, A + row_begin * K, b_wram, 1.0f,
        C + row_begin * N, rows, K, N);
  }
}
__mlu_func__ void __tl_mlu_nram_gemm_bf16_task(
    float* C, const bfloat16_t* A, const bfloat16_t* B, __wram__ bfloat16_t* b_wram,
    int32_t M, int32_t N, int32_t K, bool clear_accum, bool trans_b,
    int32_t row_begin) {
  __tl_mlu_nram_gemm_bf16_impl(C, A, B, M, N, K, clear_accum, trans_b,
                               row_begin, M, b_wram);
}
)", "NKGEMM", nram_gemm_bf16_maxnk_);
    }
    if (uses_nram_gemm_bf16_ && !nram_gemm_bf16_lean_ok_) {
      std::string text = R"(

// bfloat16 A and B. __bang_matmul writes float and does not accumulate, so a
// non-clear product lands in c_part and is added into the float C tile.
__mlu_func__ void __tl_mlu_nram_gemm_bf16_impl(float* C, const bfloat16_t* A,
                                               const bfloat16_t* B, int32_t M,
                                               int32_t N, int32_t K,
                                               bool clear_accum, bool trans_b,
                                               int32_t row_begin, int32_t rows,
                                               __wram__ bfloat16_t* b_wram,
                                               __nram__ bfloat16_t* a_panel,
                                               __nram__ bfloat16_t* b_nk,
                                               float* c_part) {
  constexpr int32_t kTile = 64;
  // Full-size A/C fragments keep the legacy per-IPU panel base (the
  // default); kernels that realize the quarter-size per-IPU allocation
  // pass row_begin = 0 explicitly.
  if (__is_mpu()) {
    return;
  }
  if ((int64_t)N * K <= NKGEMM && (clear_accum || rows * N <= 8192)) {
    // Single shot: one matmul covers the whole tile and every operand is
    // dense, so no per-row packing is needed. QK (trans_b) already stores B
    // as [N, K]; its filter reshapes straight into WRAM.
    if (clear_accum) {
      __bang_write_value(C + row_begin * N, rows * N, 0.0f);
    }
    if (trans_b) {
      __bang_reshape_filter(b_wram, B, N, 1, 1, K, NRAM2WRAM);
    } else {
      // Do NOT stage B through an NRAM2NRAM __memcpy first: that direction
      // is pathologically slow on mtp_592. a_panel and B are disjoint, so
      // transpose straight from B into the panel scratch.
      __bang_transpose(a_panel, B, K, N);
      // The transpose runs on the move queue; the reshape below consumes it
      // on the compute queue, so wait for the move queue only. Draining the
      // compute queue here would also wait on in-flight matmuls and break
      // cross-iteration pipelining.
      __sync_move();
      __bang_reshape_filter(b_wram, a_panel, N, 1, 1, K, NRAM2WRAM);
    }
    __sync_move();
    if (clear_accum) {
      __bang_matmul(C + row_begin * N, A + row_begin * K, b_wram, rows, K, N);
    } else {
      __bang_matmul(c_part, A + row_begin * K, b_wram, rows, K, N);
      __bang_add(C + row_begin * N, C + row_begin * N, c_part, rows * N);
    }
    return;
  }
  for (int32_t n0 = 0; n0 < N; n0 += kTile) {
    for (int32_t k0 = 0; k0 < K; k0 += kTile) {
      if (clear_accum && k0 == 0) {
        for (int32_t row = 0; row < rows; ++row) {
          __bang_write_value(C + (row_begin + row) * N + n0, kTile, 0.0f);
        }
      }
      if (trans_b) {
        for (int32_t column = 0; column < kTile; ++column) {
          __memcpy(b_nk + column * kTile, B + (n0 + column) * K + k0,
                   kTile * sizeof(bfloat16_t), NRAM2NRAM);
        }
      } else {
        for (int32_t k = 0; k < kTile; ++k) {
          __memcpy(a_panel + k * kTile, B + (k0 + k) * N + n0,
                   kTile * sizeof(bfloat16_t), NRAM2NRAM);
        }
        __bang_transpose(b_nk, a_panel, kTile, kTile);
        // Move-queue transpose consumed on the compute queue: wait for the
        // move queue, not the compute queue. Draining compute would also
        // wait on in-flight matmuls and break cross-iteration pipelining.
        __sync_move();
      }
      __bang_reshape_filter(a_panel, b_nk, kTile, 1, 1, kTile);
      __memcpy(b_wram, a_panel, kTile * kTile * sizeof(bfloat16_t), NRAM2WRAM);
      for (int32_t row = 0; row < rows; ++row) {
        __memcpy(a_panel + row * kTile, A + (row_begin + row) * K + k0,
                 kTile * sizeof(bfloat16_t), NRAM2NRAM);
      }
      __bang_matmul(c_part, a_panel, b_wram, rows, kTile, kTile);
      for (int32_t row = 0; row < rows; ++row) {
        float* sum = C + (row_begin + row) * N + n0;
        __bang_add(sum, sum, c_part + row * kTile, kTile);
      }
    }
  }
}
__mlu_func__ void __tl_mlu_nram_gemm_bf16_task(
    float* C, const bfloat16_t* A, const bfloat16_t* B, __wram__ bfloat16_t* b_wram,
    __nram__ bfloat16_t* a_panel, __nram__ bfloat16_t* b_nk, float* c_part,
    int32_t M, int32_t N, int32_t K, bool clear_accum, bool trans_b,
    int32_t row_begin) {
  __tl_mlu_nram_gemm_bf16_impl(C, A, B, M, N, K, clear_accum, trans_b, row_begin,
                               M, b_wram, a_panel, b_nk, c_part);
}
)";
      os << ExpandTag(text, "NKGEMM", nram_gemm_bf16_maxnk_);
    }
    if (uses_nram_gemm_) {
      std::string text = R"(

// One cluster tile. Four IPUs each own M/4 contiguous rows of A and C.
// The running sum stays in C. b_nk holds the filter, then the matmul output
// after that filter has been copied to WRAM. a_panel holds A, and also the
// [K, N] pack when B is not already [N, K].
__mlu_func__ void __tl_mlu_nram_gemm_impl(float* C, const float* A, const float* B,
                                          int32_t M, int32_t N, int32_t K,
                                          bool clear_accum, bool trans_b,
                                          int32_t row_begin, int32_t rows) {
  constexpr int32_t kTile = 64;
  __nram__ float a_panel[NKGEMM];
  __nram__ float b_nk[NKGEMM];
  __nram__ float c_part[4096];
  __wram__ float b_wram[NKGEMM];
  // Full-size A/C fragments keep the legacy per-IPU panel base (the
  // default); kernels that realize the quarter-size per-IPU allocation
  // pass row_begin = 0 explicitly.
  if (__is_mpu()) {
    return;
  }
  // The f32 single-shot path has not been validated numerically; the tiled
  // path below carries the proven 64x64 fast branch.
  if (false && (int64_t)N * K <= NKGEMM && (clear_accum || rows * N <= 8192)) {
    // Single shot: one matmul covers the whole tile and every operand is
    // dense, so no per-row packing is needed. QK (trans_b) already stores B
    // as [N, K]; its filter reshapes straight into WRAM.
    if (clear_accum) {
      __bang_write_value(C + row_begin * N, rows * N, 0.0f);
    }
    if (trans_b) {
      __bang_reshape_filter(b_wram, B, N, 1, 1, K, NRAM2WRAM);
    } else {
      __memcpy(b_nk, B, (size_t)K * N * sizeof(float), NRAM2NRAM);
      __bang_transpose(a_panel, b_nk, K, N);
      __sync_move();
      __bang_reshape_filter(b_wram, a_panel, N, 1, 1, K, NRAM2WRAM);
    }
    __sync_move();
    if (clear_accum) {
      __bang_matmul(C + row_begin * N, A + row_begin * K, b_wram, rows, K, N);
    } else {
      __bang_matmul(c_part, A + row_begin * K, b_wram, rows, K, N);
      __bang_add(C + row_begin * N, C + row_begin * N, c_part, rows * N);
    }
    return;
  }
  for (int32_t n0 = 0; n0 < N; n0 += kTile) {
    for (int32_t k0 = 0; k0 < K; k0 += kTile) {
      if (clear_accum && k0 == 0) {
        for (int32_t row = 0; row < rows; ++row) {
          __bang_write_value(C + (row_begin + row) * N + n0, kTile, 0.0f);
        }
      }
      if (trans_b) {
        for (int32_t column = 0; column < kTile; ++column) {
          __memcpy(b_nk + column * kTile, B + (n0 + column) * K + k0,
                   kTile * sizeof(float), NRAM2NRAM);
        }
      } else {
        for (int32_t k = 0; k < kTile; ++k) {
          __memcpy(a_panel + k * kTile, B + (k0 + k) * N + n0,
                   kTile * sizeof(float), NRAM2NRAM);
        }
        __bang_transpose(b_nk, a_panel, kTile, kTile);
        // Move-queue transpose consumed on the compute queue: wait for the
        // move queue, not the compute queue. Draining compute would also
        // wait on in-flight matmuls and break cross-iteration pipelining.
        __sync_move();
      }
      // Same filter layout as the global GEMM path, which matches
      // __bang_matmul. The filter has moved to WRAM, so b_nk is free to
      // receive the matmul result.
      __bang_reshape_filter(a_panel, b_nk, kTile, 1, 1, kTile);
      __memcpy(b_wram, a_panel, kTile * kTile * sizeof(float), NRAM2WRAM);
      for (int32_t row = 0; row < rows; ++row) {
        __memcpy(a_panel + row * kTile, A + (row_begin + row) * K + k0,
                 kTile * sizeof(float), NRAM2NRAM);
      }
      __bang_matmul(b_nk, a_panel, b_wram, rows, kTile, kTile);
      for (int32_t row = 0; row < rows; ++row) {
        float* sum = C + (row_begin + row) * N + n0;
        __bang_add(sum, sum, b_nk + row * kTile, kTile);
      }
    }
  }
}
__mlu_func__ void __tl_mlu_nram_gemm_task(float* C, const float* A,
                                          const float* B, __wram__ float* b_wram,
                                          int32_t M, int32_t N, int32_t K,
                                          bool clear_accum, bool trans_b,
                                          int32_t row_begin) {
  (void)b_wram;
  __tl_mlu_nram_gemm_impl(C, A, B, M, N, K, clear_accum, trans_b, row_begin, M);
}
)";
      os << ExpandTag(text, "NKGEMM", nram_gemm_maxnk_);
    }
  }

 private:
  static std::string ExpandTag(std::string text, const std::string &tag,
                               int value) {
    const std::string replacement = std::to_string(value);
    size_t pos = 0;
    while ((pos = text.find(tag, pos)) != std::string::npos) {
      text.replace(pos, tag.size(), replacement);
      pos += replacement.size();
    }
    return text;
  }

  static bool PointerElementIsBf16(const tvm::PrimExpr &expr) {
    const auto *call = expr.as<tirx::CallNode>();
    if (call == nullptr || call->args.empty()) {
      return false;
    }
    const auto *load = call->args[0].as<tirx::BufferLoadNode>();
    return load != nullptr && load->buffer->dtype.is_bfloat16();
  }

  bool uses_bang_gemm_{false};
  bool uses_nram_gemm_{false};
  bool uses_nram_gemm_bf16_{false};
  // True while every bf16 nram-gemm call site is a trans_b single shot; the
  // lean helper then omits the a_panel/b_nk staging arrays (64 KB NRAM).
  bool nram_gemm_bf16_lean_ok_{true};
  bool uses_transpose_{false};
  // Largest N*K over the nram-gemm call sites; sizes the helper's staging
  // arrays so the single-shot path can cover every call in this kernel.
  mutable int nram_gemm_maxnk_{4096};
  mutable int nram_gemm_bf16_maxnk_{4096};
  bool uses_elementwise_{false};
  bool uses_reduce_max_{false};
  bool uses_reduce_sum_{false};
  bool cluster_program_{false};
};

namespace {

class BangGemmFinder : public tirx::StmtExprVisitor {
 public:
  bool found{false};
  // Largest tile M over the gemm call sites: M <= 64 marks one-tile-per-IPU
  // kernels (flat task grid, no cluster row split).
  int max_m{0};

  void VisitExpr_(const tirx::CallNode *op) final {
    const auto *callee = op->op.as<OpNode>();
    if (callee != nullptr && (callee->name == "tl.mlu_bang_gemm" || callee->name == "tl.mlu_nram_gemm")) {
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

ffi::Module BuildTileLangMLU(IRModule mod, Target target, bool compile_device) {
  CodeGenBANG cg;
  cg.Init(/*output_ssa=*/false);

  ffi::Array<ffi::String> names;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenBANG: can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto func = Downcast<PrimFunc>(kv.second);
    BangGemmFinder finder;
    finder(func->body);
    cg.SetClusterProgram(finder.found && finder.max_m > 64);
    auto calling_conv = func->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenBANG: " << gvar << " is not a device kernel";
    cg.AddFunction(gvar, func);
    if (auto symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      names.push_back(symbol.value());
    }
  }
      std::string code = cg.Finish();
  if (compile_device) {
    const auto compile =
        ffi::Function::GetGlobal("tilelang_callback_mlu_compile");
    ICHECK(compile) << "tilelang_callback_mlu_compile is not registered";
    tvm::transform::PassContext pass_ctx = tvm::transform::PassContext::Current();
    std::string object =
        (*compile)(code, target, pass_ctx->config).cast<std::string>();
    ICHECK(!object.empty()) << "cncc produced an empty object\n" << code;
    return runtime::MLUModuleCreate(mod, code);
  }
  return CSourceModuleCreate(code, "mlu", names);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_mlu",
           [](IRModule mod, Target target) {
             return BuildTileLangMLU(mod, target, true);
           })
      .def("target.build.tilelang_mlu_without_compile",
           [](IRModule mod, Target target) {
             return BuildTileLangMLU(mod, target, false);
           });
}

} // namespace codegen
} // namespace tvm
