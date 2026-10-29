// SPDX-License-Identifier: Apache-2.0
// Ascend needs NPUStorageImpl/StorageDesc, which generic stable from_blob does
// not construct. Isolate the version-coupled torch_npu factory in this TU.
#if defined(KVCACHED_USE_ASCEND)
#ifdef TORCH_STABLE_ONLY
#undef TORCH_STABLE_ONLY
#endif
#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch_npu/csrc/aten/common/from_blob.h>

namespace kvcached {
AtenTensorHandle ascend_from_blob(void *data, int64_t count, int dtype,
                                 int device_index) {
  auto options = at::TensorOptions()
                     .dtype(static_cast<at::ScalarType>(dtype))
                     .device(c10::Device(c10::DeviceType::PrivateUse1,
                                         static_cast<c10::DeviceIndex>(device_index)));
  auto tensor = at_npu::native::from_blob(data, {count}, {1}, options);
  // Same ownership convention as the AOTI C shim. The stable wrapper deletes
  // this Tensor object; external VMM storage remains owned by FTensor.
  return reinterpret_cast<AtenTensorHandle>(new at::Tensor(std::move(tensor)));
}
} // namespace kvcached
#endif
