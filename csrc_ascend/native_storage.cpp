// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

// Optional, torch/torch_npu-version-specific bridge. The VMM allocator itself
// remains in the stable-ABI extension. No device memory is allocated or copied.
#include <torch/extension.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>

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
  // Native empty dispatch constructs the backend's TensorImpl and StorageImpl;
  // the latter carries format metadata absent from stable-ABI from_blob.
  auto result = at::empty({0}, source.options());
  auto* storage = result.storage().unsafeGetStorageImpl();
  storage->set_data_ptr(std::move(data));
  storage->set_nbytes(source.nbytes());
  storage->set_resizable(false);
  result.unsafeGetTensorImpl()->set_sizes_contiguous(source.sizes());
  // These public header fields are intentionally version-specific. The helper
  // methods are not exported by torch_npu; no hidden symbols are required.
  auto& desc = static_cast<torch_npu::NPUStorageImpl*>(storage)->npu_desc_;
  desc.base_sizes_.assign(result.sizes().begin(), result.sizes().end());
  desc.base_strides_.assign(result.strides().begin(), result.strides().end());
  desc.storage_sizes_ = desc.base_sizes_;
  desc.base_offset_ = 0;
  desc.origin_format_ = ACL_FORMAT_ND;
  desc.npu_format_ = ACL_FORMAT_ND;
  desc.data_type_ = result.dtype();
  TORCH_CHECK(result.data_ptr() == source.data_ptr(), "Storage bridge copied data");
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("wrap_nd_storage", &wrap_nd_storage,
             "Wrap VMM bytes in native NPU ND storage without copying");
}
