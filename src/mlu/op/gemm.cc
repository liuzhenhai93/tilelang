/*!
 * \file tl/mlu/op/gemm.cc
 * \brief MLU590 BANG GEMM instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace {

constexpr const char* kMLUBangGemm = "mlu.bang_gemm";

bool MatchMLUGemmTarget(Target target) {
  return target->kind->name == "mlu";
}

String SelectMLUGemm(const GemmNode& op, int block_size, Target target) {
  const bool fp32 = op.a_->dtype == DataType::Float(32) &&
                    op.b_->dtype == DataType::Float(32) &&
                    op.c_->dtype == DataType::Float(32);
  // __bang_matmul writes a float result for bfloat16 inputs and does not accumulate.
  const bool bf16 = op.a_->dtype.is_bfloat16() && op.b_->dtype.is_bfloat16() &&
                    op.c_->dtype == DataType::Float(32);
  ICHECK(fp32 || bf16)
      << "MLU BANG GEMM requires float32 A, B, and C, or bfloat16 A and B with "
         "a float32 accumulator";
  ICHECK(!op.transA_) << "MLU BANG GEMM does not support transpose_A";
  ICHECK(op.m_ > 0 && op.n_ > 0 && op.k_ > 0 && op.m_ % 16 == 0 &&
         op.n_ % 64 == 0 && op.k_ % 64 == 0 && op.m_ <= 256)
      << "MLU BANG GEMM requires M to be a positive multiple of 16 at most "
         "256, and N and K to be positive multiples of 64; got M="
      << op.m_ << ", N=" << op.n_ << ", K=" << op.k_;
  return kMLUBangGemm;
}

std::pair<int, int> ComputeMLUPartition(const GemmWarpPolicyNode& policy,
                                        int M, int N, int block_size,
                                        Target target, String gemm_inst) {
  policy.m_warp = 1;
  policy.n_warp = 1;
  return {1, 1};
}

bool RegisterMLUGemm() {
  RegisterGemmImpl(GemmImpl{"mlu.Gemm", MatchMLUGemmTarget, SelectMLUGemm,
                            ComputeMLUPartition,
                            [](String) { return false; }});
  return true;
}

const bool mlu_gemm_registered = RegisterMLUGemm();

}  // namespace
}  // namespace tl
}  // namespace tvm
