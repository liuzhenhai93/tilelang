"""MLU backend manifest and the cncc compile callback.

The pass pipeline lowers frontend IR. Device codegen emits BANG C and, when
compilation is requested, ``tilelang_callback_mlu_compile`` runs
``cncc -c -fPIC -O3 -std=c++11 --bang-mlu-arch=mtp_592``.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess

from tilelang.backend.execution_backend import ExecutionBackendSpec
from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend

from . import codegen, pipeline

_ARCH = re.compile(r"mtp_\d+")


def find_cncc() -> str:
    """Return ``cncc`` from ``NEUWARE_HOME/bin`` or ``PATH``."""
    neuware_home = os.environ.get("NEUWARE_HOME", "")
    if neuware_home:
        candidate = os.path.join(neuware_home, "bin", "cncc")
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("cncc")
    if found is not None:
        return found
    raise RuntimeError("Cannot find cncc. Install CNToolkit and put it on PATH, or set NEUWARE_HOME.")


def _hoist_shared_to_file_scope(code: str) -> str:
    """Move __mlu_shared__ declarations to file scope.

    The official MLU-OPS kernels declare their SRAM pools at file scope;
    a function-scoped __mlu_shared__ array compiles but its address window
    is not mapped at launch (MEMORY_ILLEGAL on any access), so hoist the
    declarations next to the includes before handing the source to cncc.
    """
    import os
    if os.environ.get("TL_MLU_SHARED_FILE_SCOPE", "") != "1":
        return code
    lines = code.split("\n")
    decls, body = [], []
    for ln in lines:
        if ln.lstrip().startswith("__mlu_shared__"):
            decls.append(ln)
        else:
            body.append(ln)
    if not decls:
        return code
    out = []
    inserted = False
    for ln in body:
        out.append(ln)
        if not inserted and ln.startswith("#include"):
            out.extend(decls)
            inserted = True
    if not inserted:
        return "\n".join(decls) + "\n" + "\n".join(body)
    return "\n".join(out)


def compile_mlu(code, target, pass_config=None) -> bytearray:
    """Compile BANG C source to a relocatable object. This is the MLU compile callback."""
    from tilelang.env import env
    from tilelang.transform import PassConfigKey
    from tvm.base import py_str
    from tvm.contrib import utils

    attrs = getattr(target, "attrs", None)
    arch = "mtp_592"
    if attrs is not None:
        for name in ("mcpu", "arch"):
            value = attrs.get(name)
            if value:
                arch = str(value)
                break
    if _ARCH.fullmatch(arch) is None:
        raise ValueError(f"MLU arch must look like mtp_592, got {arch!r}")

    extra: list[str] = []
    flags = (pass_config or {}).get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS)
    if isinstance(flags, str):
        extra = shlex.split(flags)
    elif flags:
        for flag in flags:
            extra.extend(shlex.split(str(flag)))

    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    source_path = temp.relpath("kernel.mlu")
    object_path = temp.relpath("kernel.o")
    code = _hoist_shared_to_file_scope(code)

    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write(code)

    command = [find_cncc(), "-c", "-fPIC", "-O3", "-std=c++11", f"--bang-mlu-arch={arch}",
               "-Xbang-cnas", "-fno-srefine", *extra]
    include = os.path.join(os.environ.get("NEUWARE_HOME", ""), "include")
    if os.path.isdir(include):
        command.append("-I" + include)
    command.extend([source_path, "-o", object_path])

    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = py_str(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"cncc failed.\nCommand: {shlex.join(command)}\nCompiler output:\n{output}\nSource:\n{code}")

    with open(object_path, "rb") as object_file:
        binary = bytearray(object_file.read())
    if not binary:
        raise RuntimeError("cncc produced an empty object file")
    return binary


BACKEND = register_backend(
    BackendModule(
        name="mlu",
        target_kinds=("mlu",),
        pipelines={"mlu": pipeline.mlu_pipeline},
        device_codegens={"mlu": codegen.MLU_CODEGEN},
        execution_backends=(ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),),
        host_codegens=STANDARD_HOST_CODEGENS,
        callbacks={"tilelang_callback_mlu_compile": compile_mlu},
    )
)
