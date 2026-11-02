"""MLU device codegen.

Both entries call the native BANG emitter. ``build`` also runs ``cncc`` through
``tilelang_callback_mlu_compile``. ``build_without_compile`` returns the source.
"""

from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen

MLU_CODEGEN = DeviceCodegen(
    "mlu",
    build=global_func_device_codegen("target.build.tilelang_mlu"),
    build_without_compile=global_func_device_codegen("target.build.tilelang_mlu_without_compile"),
)
