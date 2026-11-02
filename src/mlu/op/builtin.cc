/*!
 * \file tl/mlu/op/builtin.cc
 * \brief MLU-specific TileLang intrinsic registrations.
 */

#include <tvm/tirx/op.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TIR_DEFINE_TL_BUILTIN(mlu_bang_gemm)
    .set_num_inputs(7)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(mlu_nram_gemm)
    .set_num_inputs(8)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

}  // namespace tl
}  // namespace tvm
