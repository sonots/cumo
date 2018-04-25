#ifndef CUMO_CUDA_MEMORY_POOL_IMPL_H
#define CUMO_CUDA_MEMORY_POOL_IMPL_H

#include <algorithm>
#include <cassert>
#include <memory>
#include <mutex>
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
private:
    cudaError_t status_;

public:
    CUDARuntimeError(cudaError_t status) :
        runtime_error(cudaGetErrorString(status)), status_(status) {}
    cudaError_t status() const { return status_; }
};


class OutOfMemoryError : public std::runtime_error {
public:
    OutOfMemoryError(size_t size, size_t total) :
        runtime_error("out of memory to allocate " + std::to_string(size) + " bytes (total " + std::to_string(total) + " bytes)") {}
};

void CheckStatus(cudaError_t status);

// Memory allocation on a CUDA device.
//
// This class provides an RAII interface of the CUDA memory allocation.
class Memory {
private:
    // Pointer to the place within the buffer.
    void* ptr_ = nullptr;
    // Size of the memory allocation in bytes.
    size_t size_ = 0;
    // GPU device id whose memory the pointer refers to.
    int device_id_ = -1;

public:
    Memory(size_t size);

    ~Memory();

    intptr_t ptr() const { return reinterpret_cast<intptr_t>(ptr_); }

    size_t size() const { return size_; }

    int device_id() const { return device_id_; }
};

// A chunk points to a device memory.
//
// A chunk might be a splitted memory block from a larger allocation.
// The prev/next pointers contruct a doubly-linked list of memory addresses
// sorted by base address that must be contiguous.
class Chunk {
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
    // chunk is in use
    bool in_use_ = false;

public:
    Chunk() {}

    // mem: The device memory buffer.
    // offset: An offset bytes from the head of the buffer.
    // size: Chunk size in bytes.
    // stream_ptr: Raw stream handle of cuda stream
    Chunk(const std::shared_ptr<Memory>& mem, size_t offset, size_t size, cudaStream_t stream_ptr = 0) :
        mem_(mem), ptr_(mem->ptr() + offset), offset_(offset), size_(size), device_id_(mem->device_id()), stream_ptr_(stream_ptr) {
        assert(mem->ptr() > 0 || offset == 0);
    }

    Chunk(const Chunk&) = default;

    ~Chunk() {
        // std::cout << "Chunk dtor " << (void*)ptr_ << " " << this << std::endl;
    }

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
    friend std::shared_ptr<Chunk> Split(std::shared_ptr<Chunk>& self, size_t size);

    // Merge previously splitted block (chunk)
    friend void Merge(std::shared_ptr<Chunk>& self, std::shared_ptr<Chunk> remaining);
};

using FreeList = std::vector<std::shared_ptr<Chunk>>;  // list of free chunk
using Arena = std::vector<FreeList>;  // free_list w.r.t arena index
using ArenaIndexMap = std::vector<int>;  // arena index <=> bin size index

// Memory pool implementation for single device.
// - The allocator attempts to find the smallest cached block that will fit
//   the requested size. If the block is larger than the requested size,
//   it may be split. If no block is found, the allocator will delegate to
//   cudaMalloc.
// - If the cudaMalloc fails, the allocator will free all cached blocks that
//   are not split and retry the allocation.
class SingleDeviceMemoryPool {
private:
    int device_id_;
    std::unordered_map<intptr_t, std::shared_ptr<Chunk>> in_use_; // ptr => Chunk
    std::unordered_map<cudaStream_t, Arena> free_;
    std::unordered_map<cudaStream_t, ArenaIndexMap> index_;
    std::recursive_mutex mutex_;

public:
    SingleDeviceMemoryPool() {
        CheckStatus(cudaGetDevice(&device_id_));
    }

    intptr_t Malloc(size_t size, cudaStream_t stream_ptr = 0);

    void Free(intptr_t ptr, cudaStream_t stream_ptr = 0);

    // Free all **non-split** chunks in all arenas
    void FreeAllBlocks();

    // Free all **non-split** chunks in specified arena
    void FreeAllBlocks(cudaStream_t stream_ptr);

    size_t GetNumFreeBlocks();

    size_t GetUsedBytes();

    size_t GetFreeBytes();

    size_t GetTotalBytes() {
        return GetUsedBytes() + GetFreeBytes();
    }

// private:

    // Rounds up the memory size to fit memory alignment of cudaMalloc.
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

    bool HasArena(cudaStream_t stream_ptr) {
        auto it = free_.find(stream_ptr);
        return it != free_.end();
    }

    // Returns appropriate arena (list of bins) of a given stream.
    //
    // All free chunks in the stream belong to one of the bin in the arena.
    //
    // Caller is responsible to acquire lock.
    Arena& GetArena(cudaStream_t stream_ptr) {
        return free_[stream_ptr];  // find or create
    }

    // Returns appropriate arena sparse index of a given stream.
    //
    // Each element of the returned vector is an index value of the arena
    // for the stream. The k-th element of the arena index is the bin index
    // of the arena. For example, when the arena index is `[1, 3]`, it means
    // that the arena has 2 bins, and `arena[0]` is for bin index 1 and
    // `arena[1]` is for bin index 3.
    //
    // Caller is responsible to acquire lock.
    ArenaIndexMap& GetArenaIndexMap(cudaStream_t stream_ptr) {
        return index_[stream_ptr];  // find or create
    }

    std::shared_ptr<Chunk> PopFromFreeList(FreeList& free_list) {
        auto data = free_list.back();
        free_list.pop_back();
        return data;
    }

    // std::vector erase-remove idiom
    // http://minus9d.hatenablog.com/entry/20120605/1338896754
    bool EraseFromFreeList(FreeList& free_list, const std::shared_ptr<Chunk>& chunk) {
        assert(!chunk->in_use());
        auto iter = std::find(free_list.begin(), free_list.end(), chunk);
        if (iter == free_list.end()) {
            return false;
        }
        free_list.erase(iter);
        return true;
    }

    void AppendToFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr = 0);

    // Removes the chunk from the free list.
    //
    // @return true if the chunk can successfully be removed from
    //         the free list. false` otherwise (e.g., the chunk could not
    //         be found in the free list as the chunk is allocated.)
    bool RemoveFromFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr = 0);

    void CompactIndex(cudaStream_t stream_ptr, bool free);
};

// Memory pool for all GPU devices on the host.
//
// A memory pool preserves any allocations even if they are freed by the user.
// Freed memory buffers are held by the memory pool as *free blocks*, and they
// are reused for further memory allocations of the same sizes. The allocated
// blocks are managed for each device, so one instance of this class can be
// used for multiple devices.
// .. note::
//    When the allocation is skipped by reusing the pre-allocated block, it
//    does not call ``cudaMalloc`` and therefore CPU-GPU synchronization does
//    not occur. It makes interleaves of memory allocations and kernel
//    invocations very fast.
// .. note::
//    The memory pool holds allocated blocks without freeing as much as
//    possible. It makes the program hold most of the device memory, which may
//    make other CUDA programs running in parallel out-of-memory situation.
class MemoryPool {
private:
    int device_id() {
        int device_id = -1;
        CheckStatus(cudaGetDevice(&device_id));
        return device_id;
    }

    std::unordered_map<int, SingleDeviceMemoryPool> pools_;

public:
    MemoryPool() {}

    ~MemoryPool() { pools_.clear(); }

    // Allocates the memory, from the pool if possible.
    //
    // Args:
    //     size (int): Size of the memory buffer to allocate in bytes.
    //     stream_ptr (cudaStream_t): Get the memory from the arena of given stream
    // Returns:
    //     intptr_t: Pointer address to the allocated buffer.
    intptr_t Malloc(size_t size, cudaStream_t stream_ptr = 0) {
        auto& mp = pools_[device_id()];
        return mp.Malloc(size, stream_ptr);
    }

    // Frees the memory, to the pool
    //
    // Args:
    //     ptr (intptr_t): Pointer of the memory buffer
    //     stream_ptr (cudaStream_t): Return the memory to the arena of given stream
    void Free(intptr_t ptr, cudaStream_t stream_ptr = 0) {
        auto& mp = pools_[device_id()];
        mp.Free(ptr, stream_ptr);
    }

    // Free all **non-split** chunks in all arenas
    void FreeAllBlocks() {
        auto& mp = pools_[device_id()];
        return mp.FreeAllBlocks();
    }

    // Free all **non-split** chunks in specified arena
    //
    // Args:
    //     stream_ptr (cudaStream_t): Release free blocks in the arena of given stream
    void FreeAllBlocks(cudaStream_t stream_ptr) {
        auto& mp = pools_[device_id()];
        return mp.FreeAllBlocks(stream_ptr);
    }

    // Count the total number of free blocks.
    //
    // Returns:
    //     size_t: The total number of free blocks.
    size_t GetNumFreeBlocks() {
        auto& mp = pools_[device_id()];
        return mp.GetNumFreeBlocks();
    }

    // Get the total number of bytes used.
    //
    // Returns:
    //     size_t: The total number of bytes used.
    size_t GetUsedBytes() {
        auto& mp = pools_[device_id()];
        return mp.GetUsedBytes();
    }

    // Get the total number of bytes acquired but not used in the pool.
    //
    // Returns:
    //     size_t: The total number of bytes acquired but not used in the pool.
    size_t GetFreeBytes() {
        auto& mp = pools_[device_id()];
        return mp.GetFreeBytes();
    }

    // Get the total number of bytes acquired in the pool.
    //
    // Returns:
    //     size_t: The total number of bytes acquired in the pool.
    size_t GetTotalBytes() {
        auto& mp = pools_[device_id()];
        return mp.GetTotalBytes();
    }
};

} // namespace internal
} // namespace cumo

#endif /* ifndef CUMO_CUDA_MEMORY_POOL_IMPL_H */
