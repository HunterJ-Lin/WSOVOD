// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>
#include "ROILoopPool/ROILoopPool.h"
#include "csc/csc.h"

namespace wsovod {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_loop_pool_forward", &ROILoopPool_forward, "ROILoopPool_forward");
  m.def("roi_loop_pool_backward", &ROILoopPool_backward, "ROILoopPool_backward");
  m.def("csc_forward", &csc_forward, "csc_forward");
}

} // namespace wsovod
