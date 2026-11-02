# MLU backend source list.
#
# The target kind has no CNToolkit dependency. Compile it in every build so
# Target("mlu") exists before BANG codegen and the CNRT runtime land.
list(APPEND TILE_LANG_SRCS
  src/mlu/target.cc
  src/mlu/codegen/codegen_bang.cc
  src/mlu/op/builtin.cc
  src/mlu/op/gemm.cc
  src/mlu/op/simt.cc
  src/mlu/op/transpose.cc
  src/mlu/runtime/mlu_module.cc
)
