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
 * \file tl/mlu/target.cc
 * \brief Cambricon MLU target kind. MLU590 is BANG arch mtp_592.
 *
 * Listed from src/mlu/CMakeLists.txt in every build. Device code is BANG C
 * (cncc), not CUDA and not AscendC. The numbers below are the ones mlu-ops
 * hard-codes for this arch; NRAM, WRAM and SRAM capacities stay compiler
 * macros (__MLU_*_SIZE__, in KiB). The byte defaults below were read from
 * MLU590-M9DG via CN_DEVICE_ATTRIBUTE_* (isa 592, 8 clusters).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

#include "dlpack/dlpack.h"

namespace tvm {

namespace refl = tvm::ffi::reflection;

TVM_REGISTER_TARGET_KIND("mlu", kDLExtDev)
    // cncc --bang-mlu-arch. mtp_592 is MLU590 (mluOpDevType MLUOP_MLU590).
    .add_attr_option<ffi::String>("mcpu", refl::DefaultValue("mtp_592"))
    .add_attr_option<ffi::String>("arch", refl::DefaultValue("mtp_592"))
    // __BANG_ARCH__ for mtp_592.
    .add_attr_option<int64_t>("bang_arch", refl::DefaultValue(592))
    // IPU cores per cluster. The MPU joins __sync_cluster but is not a
    // compute core (mlu-ops CORE_DIM, arch != 322). M9DG reports 4.
    .add_attr_option<int64_t>("core_dim", refl::DefaultValue(4))
    // MLU590-M9DG reports MAX_CLUSTER_COUNT 8. Union16 is still the widest
    // job class mlu-ops can launch; the live count comes from CNRT.
    .add_attr_option<int64_t>("cluster_dim", refl::DefaultValue(8))
    .add_attr_option<int64_t>("max_cluster_dim", refl::DefaultValue(16))
    // Physical per-core / per-cluster SRAM on MLU590-M9DG. Usable NRAM is
    // nram_bytes minus nram_stack_reserve (512KiB - 128KiB = 384KiB).
    .add_attr_option<int64_t>("nram_bytes", refl::DefaultValue(512 * 1024))
    .add_attr_option<int64_t>("wram_bytes", refl::DefaultValue(512 * 1024))
    .add_attr_option<int64_t>("sram_bytes", refl::DefaultValue(2 * 1024 * 1024))
    // cncc keeps this much NRAM for its stack (REM_FOR_STACK).
    .add_attr_option<int64_t>("nram_stack_reserve",
                              refl::DefaultValue(128 * 1024))
    // NFU vector chunks, WRAM alignment, and the element count __bang_*
    // ops require (NFU_ALIGN_SIZE, WRAM_ALIGN_SIZE, COMPUTE_COUNT_ALIGN).
    .add_attr_option<int64_t>("nfu_align_bytes", refl::DefaultValue(128))
    .add_attr_option<int64_t>("wram_align_bytes", refl::DefaultValue(64))
    .add_attr_option<int64_t>("compute_count_align", refl::DefaultValue(64))
    // __mlu_global__ entries return void.
    .add_attr_option<bool>("supports_kernel_status_return",
                           refl::DefaultValue(false))
    // Launch dimensions are taskDim (clusters) and a fixed coreDim.
    // coreId is not a free threadIdx the host picks per launch.
    .add_attr_option<bool>("launch_grid_only", refl::DefaultValue(true))
    .set_default_keys({"mlu"});

} // namespace tvm
