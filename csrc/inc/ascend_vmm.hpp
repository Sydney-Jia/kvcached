// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <acl/acl.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace kvcached {
namespace gpu_vmm {

using status_t = aclError;
using allocation_handle_t = aclrtDrvMemHandle;
using allocation_prop_t = aclrtPhysicalMemProp;
using access_desc_t = aclrtMemAccessDesc;

inline const char *backend_name() { return "Ascend"; }
inline bool is_success(status_t status) { return status == ACL_SUCCESS; }

inline const char *error_string(status_t status) {
  // aclGetRecentErrMsg() is thread-local but may describe an older operation.
  // Always preserve the actual returned status instead.
  thread_local char message[64];
  std::snprintf(message, sizeof(message), "AscendCL error %d", status);
  return message;
}

inline void check(status_t status, const char *operation, const char *file,
                  unsigned line) {
  if (!is_success(status)) {
    throw std::runtime_error(std::string(file) + ':' + std::to_string(line) +
                             " " + operation + " failed: " + error_string(status));
  }
}

inline status_t initialize_runtime() {
  const auto status = aclInit(nullptr);
  // torch_npu usually owns runtime initialization and finalization.
  return status == ACL_ERROR_REPEAT_INITIALIZE ? ACL_SUCCESS : status;
}

inline status_t set_device(int device) { return aclrtSetDevice(device); }
inline int current_device() {
  int32_t device = -1;
  check(aclrtGetDevice(&device), "aclrtGetDevice", __FILE__, __LINE__);
  return device;
}

// The legacy interface name denotes a physical-device identity. Ascend UUIDs
// remain stable when different processes use different visible-device indices.
inline status_t device_get_pci_bus_id(char *identifier, int length, int device) {
  if (length < 40)
    return ACL_ERROR_INVALID_PARAM;
  aclrtUuid uuid{};
  const auto status = aclrtDeviceGetUuid(device, &uuid);
  if (!is_success(status))
    return status;
  std::snprintf(identifier, length, "ascend-");
  for (size_t i = 0; i < sizeof(uuid.bytes); ++i)
    std::snprintf(identifier + 7 + 2 * i, length - 7 - 2 * i, "%02x",
                  static_cast<unsigned char>(uuid.bytes[i]));
  return ACL_SUCCESS;
}

inline status_t mem_get_info(size_t *free_bytes, size_t *total_bytes) {
  return aclrtGetMemInfo(ACL_HBM_MEM, free_bytes, total_bytes);
}
inline status_t device_is_integrated(int *integrated, int /*device*/) {
  *integrated = 0;
  return ACL_SUCCESS;
}
inline status_t device_synchronize() { return aclrtSynchronizeDevice(); }

inline allocation_prop_t make_pinned_device_allocation_prop(int device) {
  allocation_prop_t prop{};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}
inline access_desc_t make_device_rw_access_desc(int device) {
  access_desc_t desc{};
  desc.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = device;
  desc.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
  return desc;
}
inline status_t get_allocation_granularity(size_t *granularity,
                                           const allocation_prop_t *prop) {
  auto copy = *prop;
  return aclrtMemGetAllocationGranularity(
      &copy, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, granularity);
}
inline status_t get_vmm_support(int *supported, int device) {
  auto prop = make_pinned_device_allocation_prop(device);
  size_t granularity = 0;
  const auto status = get_allocation_granularity(&granularity, &prop);
  *supported = is_success(status) && granularity != 0;
  return status;
}
inline status_t mem_create(allocation_handle_t *handle, size_t size,
                           const allocation_prop_t *prop) {
  return aclrtMallocPhysical(handle, size, prop, 0);
}
inline status_t mem_release(allocation_handle_t handle) {
  return aclrtFreePhysical(handle);
}
inline status_t address_reserve(void **ptr, size_t size, size_t alignment,
                                void *preferred_addr = nullptr) {
  return aclrtReserveMemAddress(ptr, size, alignment, preferred_addr, 0);
}
inline status_t address_free(void *ptr, size_t /*size*/) {
  return aclrtReleaseMemAddress(ptr);
}
inline status_t mem_map(void *ptr, size_t size, size_t offset,
                        allocation_handle_t handle) {
  return aclrtMapMem(ptr, size, offset, handle, 0);
}
// CANN removes the complete mapping beginning at ptr. Callers must pass the
// original mapping boundary, not an arbitrary CUDA-style subrange.
inline status_t mem_unmap(void *ptr, size_t /*size*/) {
  return aclrtUnmapMem(ptr);
}
inline status_t set_access(void * /*ptr*/, size_t /*size*/,
                           const access_desc_t *desc, size_t count) {
  // aclrtMapMem grants access on the allocating device. aclrtMemSetAccess is
  // for peer-device access, which this backend does not silently enable.
  if (count != 1 || desc[0].location.type != ACL_MEM_LOCATION_TYPE_DEVICE ||
      desc[0].location.id != static_cast<uint32_t>(current_device()) ||
      desc[0].flags != ACL_RT_MEM_ACCESS_FLAGS_READWRITE)
    return ACL_ERROR_INVALID_PARAM;
  return ACL_SUCCESS;
}

} // namespace gpu_vmm
} // namespace kvcached
