// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Optional, torch/torch_npu-version-specific bridge. The VMM allocator itself
// remains in the stable-ABI extension. No device memory is allocated or copied.
#include <torch/extension.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/framework/StorageDescHelper.h>

at::Tensor wrap_nd_storage(const at::Tensor& source) {
  TORCH_CHECK(source.device().type() == c10::DeviceType::PrivateUse1,
              "Expected an NPU tensor");
  TORCH_CHECK(source.is_contiguous() && source.storage_offset() == 0,
              "Wrap the original contiguous VMM reservation before slicing");
  TORCH_CHECK(source.scalar_type() == at::ScalarType::Char,
              "Native cache backings must be byte tensors (int8)");
  auto owner = std::make_unique<at::Tensor>(source);
  c10::DataPtr data(source.data_ptr(), owner.get(),
                    [](void* ptr) { delete static_cast<at::Tensor*>(ptr); },
                    source.device());
  owner.release();
  auto storage = c10::make_intrusive<torch_npu::NPUStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), source.nbytes(), std::move(data),
      nullptr, false);
  // A zero-element native tensor gives us the backend's normal TensorImpl.
  auto result = at::empty({0}, source.options());
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(c10::Storage(storage));
  result.unsafeGetTensorImpl()->set_sizes_contiguous(source.sizes());
  at_npu::native::StorageDescHelper::SetDesc(
      result, result.sizes(), result.strides(), ACL_FORMAT_ND);
  TORCH_CHECK(result.data_ptr() == source.data_ptr(), "Storage bridge copied data");
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("wrap_nd_storage", &wrap_nd_storage,
             "Wrap VMM bytes in native NPU ND storage without copying");
}
