/*!
 * \file tl/mlu/op/simt.cc
 * \brief Scalar MLU lowering for copy, fill, and reduce.
 *
 * Fragment buffers are IPU-private NRAM. These ops become ordinary loops.
 * The cluster row split is applied later, to the same rows the GEMM helper
 * owns.
 */

#include "backend/common/op/reduce.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/reduce.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

bool IsMLU(Target target) { return target->kind->name == "mlu"; }

bool IsPrivate(const Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  const auto scope = buffer.scope();
  return scope == "local.fragment" || scope == "local" || scope == "local.var";
}

LayoutMap InferCopyLayout(const CopyNode &op, const LayoutInferArgs &layout_args, InferLevel level) {
  (void)op;
  (void)layout_args;
  (void)level;
  return {};
}

Stmt LowerCopy(const CopyNode &op, const LowerArgs &lower_args, arith::Analyzer *analyzer) {
  (void)lower_args;
  return op.MakeSIMTLoop(analyzer);
}

Stmt LowerFill(const FillNode &op, const LowerArgs &lower_args, arith::Analyzer *analyzer) {
  (void)lower_args;
  if (!IsPrivate(op.dst) && !IsGlobalBuffer(op.dst)) {
    LOG(FATAL) << "MLU fill destination scope is `" << op.dst.scope() << "`";
  }
  return op.MakeSIMTLoop(analyzer);
}

Stmt LowerReduce(const ReduceOpNode &op, const LowerArgs &lower_args, arith::Analyzer *analyzer) {
  (void)analyzer;
  ICHECK(IsPrivate(op.src) && IsPrivate(op.dst))
      << "MLU reduce expects fragment src and dst, got `" << op.src.scope() << "` and `"
      << op.dst.scope() << "`";
  ICHECK_EQ(op.batch, 1);
  ICHECK(!op.nan_propagate);

  (void)lower_args;
  Buffer src_buffer = op.src;
  Buffer dst_buffer = op.dst;
  int src_dim = static_cast<int>(op.src->shape.size());
  int dst_dim = static_cast<int>(op.dst->shape.size());
  ICHECK_GE(op.dim, 0);
  ICHECK_LT(op.dim, src_dim);
  ICHECK(dst_dim == src_dim - 1 || dst_dim == src_dim);
  ICHECK(src_buffer->dtype == dst_buffer->dtype);

  Array<Var> dst_vars;
  Array<PrimExpr> dst_indices;
  for (int i = 0; i < dst_dim; ++i) {
    Var var("i" + std::to_string(i));
    dst_vars.push_back(var);
    dst_indices.push_back(var);
  }
  auto make_src_indices = [&](PrimExpr reduce_index) {
    Array<PrimExpr> indices;
    for (int i = 0; i < src_dim; ++i) {
      if (i == op.dim) {
        indices.push_back(reduce_index);
      } else if (dst_dim == src_dim) {
        indices.push_back(dst_vars[i]);
      } else {
        indices.push_back(dst_vars[i < op.dim ? i : i - 1]);
      }
    }
    return indices;
  };

  const auto *length = op.src->shape[op.dim].as<IntImmNode>();
  const bool vector_row =
      length != nullptr && length->value > 0 && length->value % 64 == 0 &&
      op.dim == src_dim - 1 && dst_dim == src_dim - 1 &&
      src_buffer->dtype == DataType::Float(32) && (op.type->IsMax() || op.type->IsSum());
  // A whole [rows, cols] panel reduces with ONE pool instruction instead of
  // one call per row. Measured slower than the per-row path on mtp_592 (the
  // pool engine throughput loses to 64-element __bang_max rows), so it is
  // opt-in via TL_MLU_REDUCE2D=1.
  const auto *rows_imm =
      dst_dim == 1 && getenv("TL_MLU_REDUCE2D") != nullptr
          ? op.dst->shape[0].as<IntImmNode>()
          : nullptr;
  // Axis-0 pool reduce: dst[j] = reduce over src[:, j]. Used for the
  // transposed softmax panel where each original row is a column.
  const auto *rows0 =
      src_dim == 2 && dst_dim == 1 && op.dim == 0
          ? op.src->shape[0].as<IntImmNode>()
          : nullptr;
  const auto *cols0 =
      src_dim == 2 && dst_dim == 1 && op.dim == 0
          ? op.src->shape[1].as<IntImmNode>()
          : nullptr;
  const bool vector_panel_axis0 =
      rows0 != nullptr && cols0 != nullptr && rows0->value > 0 &&
      cols0->value > 0 && cols0->value <= 256 &&
      rows0->value * cols0->value % 64 == 0 &&
      src_buffer->dtype == DataType::Float(32) &&
      (op.type->IsMax() || op.type->IsSum());
  Stmt body;
  if (vector_panel_axis0) {
    PrimExpr dst_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(dst_buffer, {IntImm(DataType::Int(32), 0)})});
    PrimExpr src_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(src_buffer,
                         {IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0)})});
    body = Evaluate(Call(
        DataType::Void(), builtin::call_extern(),
        {StringImm(op.type->IsMax() ? "__tl_mlu_reduce2da_max"
                                    : "__tl_mlu_reduce2da_sum"),
         dst_ptr, src_ptr, IntImm(DataType::Int(32), rows0->value),
         IntImm(DataType::Int(32), cols0->value),
         IntImm(DataType::Int(32), op.clear ? 1 : 0)}));
    return body;
  }
  const bool vector_panel =
      length != nullptr && rows_imm != nullptr && rows_imm->value > 0 &&
      length->value > 0 && length->value % 64 == 0 &&
      src_dim == 2 && dst_dim == 1 && op.dim == 1 &&
      src_buffer->dtype == DataType::Float(32) &&
      (op.type->IsMax() || op.type->IsSum());
  if (vector_panel) {
    PrimExpr dst_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(dst_buffer, {IntImm(DataType::Int(32), 0)})});
    PrimExpr src_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(src_buffer,
                         {IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0)})});
    body = Evaluate(Call(
        DataType::Void(), builtin::call_extern(),
        {StringImm(op.type->IsMax() ? "__tl_mlu_reduce2d_max" : "__tl_mlu_reduce2d_sum"),
         dst_ptr, src_ptr, IntImm(DataType::Int(32), rows_imm->value),
         IntImm(DataType::Int(32), length->value),
         IntImm(DataType::Int(32), op.clear ? 1 : 0)}));
    return body;
  }
  if (vector_row) {
    PrimExpr dst_ptr =
        Call(DataType::Handle(), builtin::address_of(), {BufferLoad(dst_buffer, dst_indices)});
    PrimExpr src_ptr = Call(DataType::Handle(), builtin::address_of(),
                             {BufferLoad(src_buffer, make_src_indices(IntImm(DataType::Int(32), 0)))});
    body = Evaluate(Call(DataType::Void(), builtin::call_extern(),
                          {StringImm(op.type->IsMax() ? "__tl_mlu_reduce_max" : "__tl_mlu_reduce_sum"),
                           dst_ptr, src_ptr, IntImm(DataType::Int(32), length->value),
                           IntImm(DataType::Int(32), op.clear ? 1 : 0)}));
  } else {
    Array<Stmt> stmts;
    if (op.clear) {
      stmts.push_back(BufferStore(dst_buffer, backend::reduce::MakeInitValue(op), dst_indices));
    }
    Var rv("rv");
    stmts.push_back(For(
        rv, 0, op.src->shape[op.dim], ForKind::kSerial,
        BufferStore(dst_buffer,
                    backend::reduce::MakeReduce(op, /*vsize=*/1, BufferLoad(dst_buffer, dst_indices),
                                                BufferLoad(src_buffer, make_src_indices(rv))),
                    dst_indices)));
    body = SeqStmt::Flatten(stmts);
  }
  for (int i = dst_dim - 1; i >= 0; --i) {
    body = For(dst_vars[i], 0, op.dst->shape[i], ForKind::kSerial, body);
  }
  return body;
}

bool RegisterMLUSimt() {
  RegisterCopyImpl(CopyImpl{"mlu.Copy", IsMLU, 100, InferCopyLayout, LowerCopy});
  RegisterFillImpl(FillImpl{"mlu.Fill", IsMLU, LowerFill});
  RegisterReduceImpl(ReduceImpl{"mlu.Reduce", IsMLU, LowerReduce});
  return true;
}

const bool kRegistered = RegisterMLUSimt();

}  // namespace
}  // namespace tl
}  // namespace tvm
