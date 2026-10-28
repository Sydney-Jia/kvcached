// SPDX-FileCopyrightText: Copyright contributors to the kvcached project
// SPDX-License-Identifier: Apache-2.0
// Opt-in measurement only. Forward every call, preserving arguments/results.
#include <acl/acl.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns() {
  timespec t{};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}

template <typename T> static T original(const char* name) {
  void* fn = dlsym(RTLD_NEXT, name);
  if (!fn) {
    auto handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
    if (handle) fn = dlsym(handle, name);
  }
  if (!fn) { dprintf(2, "VMM timing cannot resolve %s\n", name); _exit(127); }
  return reinterpret_cast<T>(fn);
}

static void record(const char* op, uint64_t start, uint64_t end,
                   aclError status, size_t size, uintptr_t address,
                   uintptr_t handle) {
  int saved_errno = errno;
  thread_local int fd = -1;
  thread_local pid_t owner = 0;
  pid_t pid = getpid();
  if (owner != pid) {
    if (fd >= 0) close(fd);
    owner = pid;
    const char* prefix = getenv("KVCACHED_TIMING_PREFIX");
    if (!prefix) { errno = saved_errno; return; }
    char path[4096];
    snprintf(path, sizeof(path), "%s.%d.csv", prefix, int(pid));
    fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd < 0) { dprintf(2, "VMM timing open failed\n"); _exit(127); }
  }
  if (fd >= 0) {
    char line[256];
    int n = snprintf(line, sizeof(line), "%s,%d,%ld,%llu,%llu,%d,%zu,%llx,%llx\n",
                     op, int(pid), syscall(SYS_gettid),
                     (unsigned long long)start, (unsigned long long)end,
                     int(status), size, (unsigned long long)address,
                     (unsigned long long)handle);
    if (write(fd, line, n) != n) { dprintf(2, "VMM timing write failed\n"); _exit(127); }
  }
  errno = saved_errno;
}

extern "C" aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size,
                                        const aclrtPhysicalMemProp* prop,
                                        uint64_t flags) {
  static auto fn = original<decltype(&aclrtMallocPhysical)>("aclrtMallocPhysical");
  auto start = now_ns(); auto rc = fn(handle, size, prop, flags); auto end = now_ns();
  record("alloc", start, end, rc, size, 0, rc == 0 ? (uintptr_t)*handle : 0);
  return rc;
}
extern "C" aclError aclrtFreePhysical(aclrtDrvMemHandle handle) {
  static auto fn = original<decltype(&aclrtFreePhysical)>("aclrtFreePhysical");
  auto start = now_ns(); auto rc = fn(handle); auto end = now_ns();
  record("free", start, end, rc, 0, 0, (uintptr_t)handle); return rc;
}
extern "C" aclError aclrtMapMem(void* ptr, size_t size, size_t offset,
                                aclrtDrvMemHandle handle, uint64_t flags) {
  static auto fn = original<decltype(&aclrtMapMem)>("aclrtMapMem");
  auto start = now_ns(); auto rc = fn(ptr, size, offset, handle, flags); auto end = now_ns();
  record("map", start, end, rc, size, (uintptr_t)ptr, (uintptr_t)handle); return rc;
}
extern "C" aclError aclrtUnmapMem(void* ptr) {
  static auto fn = original<decltype(&aclrtUnmapMem)>("aclrtUnmapMem");
  auto start = now_ns(); auto rc = fn(ptr); auto end = now_ns();
  record("unmap", start, end, rc, 0, (uintptr_t)ptr, 0); return rc;
}
extern "C" aclError aclrtSynchronizeDevice() {
  static auto fn = original<decltype(&aclrtSynchronizeDevice)>("aclrtSynchronizeDevice");
  auto start = now_ns(); auto rc = fn(); auto end = now_ns();
  record("sync", start, end, rc, 0, 0, 0); return rc;
}
