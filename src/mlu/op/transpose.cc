/*!
 * \file tl/mlu/op/transpose.cc
 * \brief MLU implementation for tl.transpose lowering.
 *
 * Fragment buffers are IPU-private NRAM. A dense 2D fragment transpose maps
 * to one __bang_transpose instruction; anything else falls back to the SIMT
 * element loop.
 */

#include "op/transpose.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt.h>

#include "op/utils.h"

namespace tvm {
namespace tl {
namespace mlu {

using namespace tirx;
using namespace ffi;

namespace {

bool IsMLU(Target target) { return target->kind->name == "mlu"; }

bool IsNramPrivate(const Buffer &buffer) {
  const std::string scope = buffer.scope();
  return scope == "local.fragment" || scope == "local" || scope == "local.var";
}

const IntImmNode *AsIntImm(const PrimExpr &expr) {
  return expr.as<IntImmNode>();
}

/*!
 * Dense 2D transpose: src [h, w] and dst [w, h] cover their whole buffers,
 * the streamed inner dim is NFU-aligned, and the dtype has a transpose
 * instruction. One __bang_transpose replaces the element loop.
 */
bool CanTransposeWhole(const TransposeNode &op) {
  if (!IsNramPrivate(op.src) || !IsNramPrivate(op.dst)) {
    return false;
  }
  if (op.src->shape.size() != 2 || op.dst->shape.size() != 2) {
    return false;
  }
  const int elem_bytes = (op.src->dtype.bits() + 7) / 8;
  if (elem_bytes != 4 && elem_bytes != 2 && elem_bytes != 1) {
    return false;
  }
  const auto *h = AsIntImm(op.src->shape[0]);
  const auto *w = AsIntImm(op.src->shape[1]);
  const auto *dh = AsIntImm(op.dst->shape[0]);
  const auto *dw = AsIntImm(op.dst->shape[1]);
  if (h == nullptr || w == nullptr || dh == nullptr || dw == nullptr) {
    return false;
  }
  if (h->value <= 0 || w->value <= 0 || dh->value != w->value ||
      dw->value != h->value) {
    return false;
  }
  if (op.src_range.size() != 2 || op.dst_range.size() != 2) {
    return false;
  }
  // The src region must cover the full src buffer so one instruction is
  // exact; the dst region is verified against the transposed shape below.
  for (size_t i = 0; i < 2; ++i) {
    const auto *min = op.src_range[i]->min.as<IntImmNode>();
    const auto *ext = op.src_range[i]->extent.as<IntImmNode>();
    if (min == nullptr || ext == nullptr || min->value != 0 ||
        ext->value != (i == 0 ? h->value : w->value)) {
      return false;
    }
  }
  // dst ranges mirror the transposed shape; verify against dst->shape too.
  const auto *dmin0 = op.dst_range[0]->min.as<IntImmNode>();
  const auto *dext0 = op.dst_range[0]->extent.as<IntImmNode>();
  const auto *dmin1 = op.dst_range[1]->min.as<IntImmNode>();
  const auto *dext1 = op.dst_range[1]->extent.as<IntImmNode>();
  if (dmin0 == nullptr || dmin0->value != 0 || dmin1 == nullptr ||
      dmin1->value != 0 || dext0 == nullptr || dext1 == nullptr ||
      dext0->value != dh->value || dext1->value != dw->value) {
    return false;
  }
  // The streamed inner dim must be NFU-aligned (128 bytes).
  if ((w->value * elem_bytes) % 128 != 0) {
    return false;
  }
  if (op.src->dtype != op.dst->dtype) {
    return false;
  }
  return true;
}

Stmt Lower(const TransposeNode &op, const LowerArgs &lower_args,
           arith::Analyzer *analyzer) {
  (void)lower_args;
  (void)analyzer;
  if (CanTransposeWhole(op)) {
    const auto *h = AsIntImm(op.src->shape[0]);
    const auto *w = AsIntImm(op.src->shape[1]);
    PrimExpr dst_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(op.dst, {IntImm(DataType::Int(32), 0),
                                  IntImm(DataType::Int(32), 0)})});
    PrimExpr src_ptr =
        Call(DataType::Handle(), builtin::address_of(),
             {BufferLoad(op.src, {IntImm(DataType::Int(32), 0),
                                  IntImm(DataType::Int(32), 0)})});
    return Evaluate(Call(
        DataType::Void(), builtin::call_extern(),
        {StringImm("__bang_transpose"), dst_ptr, src_ptr,
         IntImm(DataType::Int(32), h->value), IntImm(DataType::Int(32), w->value)}));
  }
  return op.MakeSIMTLoop(analyzer);
}

} // namespace mlu

namespace {

bool RegisterMLUTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "mlu.Transpose",
      mlu::IsMLU,
      mlu::Lower,
  });
  return true;
}

const bool mlu_transpose_registered = RegisterMLUTranspose();

} // namespace

} // namespace mlu
} // namespace tl
} // namespace tvm
