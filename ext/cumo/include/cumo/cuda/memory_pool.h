#ifndef CUMO_CUDA_MEMORY_POOL_H
#define CUMO_CUDA_MEMORY_POOL_H

#include <algorithm>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// CUDA memory pool implementation highly referring CuPy

namespace cumo {
namespace internal {

// cudaMalloc() is aligned to at least 512 bytes
// cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
constexpr int kRoundSize = 512; // bytes

class CUDARuntimeError : public std::runtime_error {
public:
    CUDARuntimeError(cudaError_t status) :
        runtime_error(cudaGetErrorString(status)), status_(status) {}
    cudaError_t status() const { return status_; }
private:
    cudaError_t status_;
};

void CheckStatus(cudaError_t status);

// Memory allocation on a CUDA device.
//
// This class provides an RAII interface of the CUDA memory allocation.
class Memory {
public:
    // size: Size of the memory allocation in bytes.
    Memory(size_t size) : size_(size) {
        if (size_ > 0) {
            CheckStatus(cudaGetDevice(&device_id_));
            CheckStatus(cudaMallocManaged(&ptr_, size_, cudaMemAttachGlobal));
        }
    }

    ~Memory() {
        if (size_ > 0) {
            CheckStatus(cudaFree(ptr_));
        }
    }

    // Returns the pointer value to the head of the allocation.
    intptr_t ptr() const { return reinterpret_cast<intptr_t>(ptr_); }
    size_t size() const { return size_; }
    int device_id() const { return device_id_; }
private:
    // Pointer to the place within the buffer.
    void* ptr_ = nullptr;
    // Size of the memory allocation in bytes.
    size_t size_ = 0;
    // GPU device id whose memory the pointer refers to.
    int device_id_ = -1;
};

// A chunk points to a device memory.
//
// A chunk might be a splitted memory block from a larger allocation.
// The prev/next pointers contruct a doubly-linked list of memory addresses
// sorted by base address that must be contiguous.
class Chunk : public std::enable_shared_from_this<Chunk> {
public:
    // mem: The device memory buffer.
    // offset: An offset bytes from the head of the buffer.
    // size: Chunk size in bytes.
    // stream_ptr: Raw stream handle of cuda stream
    Chunk(const std::shared_ptr<Memory>& mem, size_t offset, size_t size, cudaStream_t stream_ptr = 0) :
        mem_(mem), ptr_(mem->ptr() + offset), offset_(offset), size_(size), device_id_(mem->device_id()), stream_ptr_(stream_ptr) {
        assert(mem->ptr() > 0 || offset == 0);
    }

    Chunk(const Chunk&) = default;

    intptr_t ptr() const { return ptr_; }

    size_t offset() const { return offset_; }

    size_t size() const { return size_; }

    int device_id() const { return device_id_; }

    const std::shared_ptr<Chunk>& prev() const { return prev_; }

    std::shared_ptr<Chunk>& prev() { return prev_; }

    const std::shared_ptr<Chunk>& next() const { return next_; }

    std::shared_ptr<Chunk>& next() { return next_; }

    cudaStream_t stream_ptr() const { return stream_ptr_; }

    void set_prev(const std::shared_ptr<Chunk>& prev) { prev_ = prev; }

    void set_next(const std::shared_ptr<Chunk>& next) { next_ = next; }

    bool in_use() const { return in_use_; }

    void set_in_use(bool in_use) { in_use_ = in_use; }

    // Split contiguous block of a larger allocation
    std::shared_ptr<Chunk> Split(size_t size);

    // Merge previously splitted block (chunk)
    void Merge(std::shared_ptr<Chunk>& remaining);

private:
    // The device memory buffer.
    std::shared_ptr<Memory> mem_;
    // Memory address.
    intptr_t ptr_ = 0;
    // An offset bytes from the head of the buffer.
    size_t offset_ = 0;
    // Chunk size in bytes.
    size_t size_ = 0;
    // GPU device id whose memory the pointer refers to.
    int device_id_;
    // prev memory pointer if split from a larger allocation
    std::shared_ptr<Chunk> prev_;
    // next memory pointer if split from a larger allocation
    std::shared_ptr<Chunk> next_;
    // Raw stream handle of cuda stream
    cudaStream_t stream_ptr_;

    bool in_use_ = false;
};

// Memory pool implementation for single device.
// - The allocator attempts to find the smallest cached block that will fit
//   the requested size. If the block is larger than the requested size,
//   it may be split. If no block is found, the allocator will delegate to
//   cudaMalloc.
// - If the cudaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
// class SingleDeviceMemoryPool {
class MemoryPool {
    using FreeList = std::vector<std::shared_ptr<Chunk>>;  // list of free chunk
    using Arena = std::vector<FreeList>;  // free_list w.r.t arena index
    using ArenaIndexMap = std::vector<int>;  // arena index <=> bin size index

private:
    int device_id_;
    std::unordered_map<intptr_t, std::shared_ptr<Chunk>> in_use_; // ptr => Chunk
    std::unordered_map<cudaStream_t, Arena> free_;
    std::unordered_map<cudaStream_t, ArenaIndexMap> index_;

public:
    //SingleDeviceMemoryPool() {
    MemoryPool() {
        CheckStatus(cudaGetDevice(&device_id_));
    }

    intptr_t Malloc(size_t size);

    void Free(intptr_t ptr);

// private:

    // TODO: std::shared_ptr<Chunk>&
    void AppendToFreeList(size_t size, std::shared_ptr<Chunk> chunk, cudaStream_t stream_ptr = 0);

    bool RemoveFromFreeList(size_t size, std::shared_ptr<Chunk> chunk, cudaStream_t stream_ptr = 0);

    // Round up the memory size to fit memory alignment of cudaMalloc.
    size_t GetRoundedSize(size_t size) {
        return ((size + kRoundSize - 1) / kRoundSize) * kRoundSize;
    }

    // Get bin index regarding the memory size
    int GetBinIndex(size_t size) {
        return (size - 1) / kRoundSize;
    }

    int GetArenaIndex(size_t size, cudaStream_t stream_ptr = 0) {
        int bin_index = GetBinIndex(size);
        ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
        return std::lower_bound(arena_index_map.begin(), arena_index_map.end(), bin_index) - arena_index_map.begin();
    }

    // Get appropriate arena (list of bins) of a given stream
    Arena& GetArena(cudaStream_t stream_ptr) {
        return free_[stream_ptr];  // find or create
    }

    // Get appropriate arena sparse index of a given stream
    ArenaIndexMap& GetArenaIndexMap(cudaStream_t stream_ptr) {
        return index_[stream_ptr];  // find or create
    }

    std::shared_ptr<Chunk>& PopFromFreeList(FreeList& free_list) {
        auto& data = free_list.back();
        free_list.pop_back();
        return data;
    }

    // std::vector erase-remove idiom
    // http://minus9d.hatenablog.com/entry/20120605/1338896754
    bool EraseFromFreeList(FreeList& free_list, const std::shared_ptr<Chunk>& chunk) {
        auto iter = std::find(free_list.begin(), free_list.end(), chunk);
        if (iter == free_list.end()) {
            return false;
        }
        free_list.erase(iter);
        return true;
    }

    //TODO(sonots): Implement
    //cpdef free_all_blocks(self, stream=None):
    //    """Free all **non-split** chunks"""
    //    cdef size_t stream_ptr
    //    rlock.lock_fastrlock(self._free_lock, -1, True)
    //    try:
    //        # free blocks in all arenas
    //        if stream is None:
    //            for stream_ptr in list(self._free.iterkeys()):
    //                _compact_index(self, stream_ptr, True)
    //        else:
    //            _compact_index(self, stream.ptr, True)
    //    finally:
    //        rlock.unlock_fastrlock(self._free_lock)

    //TODO(sonots): Implement
    //cpdef n_free_blocks(self):
    //    cdef Py_ssize_t n = 0
    //    cdef set free_list
    //    rlock.lock_fastrlock(self._free_lock, -1, True)
    //    try:
    //        for arena in self._free.itervalues():
    //            for v in arena:
    //                if v is not None:
    //                    n += len(v)
    //    finally:
    //        rlock.unlock_fastrlock(self._free_lock)
    //    return n

    //TODO(sonots): Implement
    //cpdef used_bytes(self):
    //    cdef Py_ssize_t size = 0
    //    cdef _Chunk chunk
    //    rlock.lock_fastrlock(self._in_use_lock, -1, True)
    //    try:
    //        for chunk in self._in_use.itervalues():
    //            size += chunk.size
    //    finally:
    //        rlock.unlock_fastrlock(self._in_use_lock)
    //    return size

    //TODO(sonots): Implement
    //cpdef free_bytes(self):
    //    cdef Py_ssize_t size = 0
    //    cdef set free_list
    //    cdef _Chunk chunk
    //    rlock.lock_fastrlock(self._free_lock, -1, True)
    //    try:
    //        for arena in self._free.itervalues():
    //            for free_list in arena:
    //                if free_list is None:
    //                    continue
    //                for chunk in free_list:
    //                    size += chunk.size
    //    finally:
    //        rlock.unlock_fastrlock(self._free_lock)
    //    return size

    //cpdef total_bytes(self):
    //    return self.used_bytes() + self.free_bytes()
};

} // namespace internal
} // namespace cumo

#endif /* ifndef CUMO_CUDA_MEMORY_POOL_H */
