#ifndef CUMO_CUDA_MEMORY_POOL_IMPL_H
#define CUMO_CUDA_MEMORY_POOL_IMPL_H

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

// CUDA memory pool implementation highly referring CuPy

namespace cumo {
namespace internal {

// cudaMalloc() is aligned to at least 512 bytes
// cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
constexpr int kRoundSize = 512; // bytes

// The largest size which can be rounded up to a multiple of kRoundSize without
// wrapping around in size_t. Larger requests can never be satisfied anyway.
constexpr size_t kMaxAllocationSize = std::numeric_limits<size_t>::max() - (kRoundSize - 1);

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
using ArenaIndexMap = std::vector<size_t>;  // arena index <=> bin size index

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

    // Returns false if this pool did not allocate the pointer, in which case
    // nothing was freed.
    bool Free(intptr_t ptr, cudaStream_t stream_ptr = 0);

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
    //
    // `size` must not exceed kMaxAllocationSize. Otherwise the addition below
    // wraps around in size_t, and the largest sizes would round down to 0.
    size_t GetRoundedSize(size_t size) {
        assert(size <= kMaxAllocationSize);
        return ((size + kRoundSize - 1) / kRoundSize) * kRoundSize;
    }

    // Get bin index regarding the memory size
    //
    // The index is a size_t rather than an int because the bin index of a huge
    // size does not fit in an int. A truncated (possibly negative) index makes
    // the free list search below start at the wrong bin and hand out a chunk
    // smaller than requested.
    //
    // `size` must be positive; a zero-sized allocation has no bin.
    size_t GetBinIndex(size_t size) {
        assert(size > 0);
        return (size - 1) / kRoundSize;
    }

    size_t GetArenaIndex(size_t size, cudaStream_t stream_ptr = 0) {
        size_t bin_index = GetBinIndex(size);
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
    std::mutex pools_mutex_;

    // Returns the pool of the current device, creating it on first use.
    //
    // The lock only has to cover the lookup: pools_ is never erased from
    // before destruction, and unordered_map keeps references to its elements
    // valid across a rehash, so the returned reference outlives the lock.
    // Without it, the insert operator[] performs on first use could rehash the
    // map underneath another thread walking it.
    SingleDeviceMemoryPool& GetPool() {
        int id = device_id();  // a CUDA call, so keep it out of the lock
        std::lock_guard<std::mutex> lock{pools_mutex_};
        return pools_[id];  // find or create
    }

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
        return GetPool().Malloc(size, stream_ptr);
    }

    // Frees the memory, to the pool
    //
    // Args:
    //     ptr (intptr_t): Pointer of the memory buffer
    //     stream_ptr (cudaStream_t): Return the memory to the arena of given stream
    // Returns:
    //     bool: false if no pool allocated the pointer, in which case nothing
    //           was freed and the caller has to free it by its own means.
    bool Free(intptr_t ptr, cudaStream_t stream_ptr = 0) {
        int current_device_id = device_id();

        // Take the pointers out under the lock and release it before freeing:
        // an element of pools_ stays put once inserted, so the pointers remain
        // valid, and this keeps a Free of one device's pool from blocking a
        // Malloc on another. `others` does not allocate in the usual
        // single-device case, where it stays empty.
        SingleDeviceMemoryPool* current = nullptr;
        std::vector<SingleDeviceMemoryPool*> others;
        {
            std::lock_guard<std::mutex> lock{pools_mutex_};
            for (auto& entry : pools_) {
                if (entry.first == current_device_id) {
                    current = &entry.second;
                } else {
                    others.emplace_back(&entry.second);
                }
            }
        }

        if (current != nullptr && current->Free(ptr, stream_ptr)) {
            return true;
        }
        // The current device may have been switched since the allocation.
        // cudaMallocManaged hands out addresses which are unique across the
        // host and every device, so the pointer identifies its pool on its own.
        for (SingleDeviceMemoryPool* mp : others) {
            if (mp->Free(ptr, stream_ptr)) {
                return true;
            }
        }
        return false;
    }

    // Free all **non-split** chunks in all arenas
    void FreeAllBlocks() {
        return GetPool().FreeAllBlocks();
    }

    // Free all **non-split** chunks in specified arena
    //
    // Args:
    //     stream_ptr (cudaStream_t): Release free blocks in the arena of given stream
    void FreeAllBlocks(cudaStream_t stream_ptr) {
        return GetPool().FreeAllBlocks(stream_ptr);
    }

    // Count the total number of free blocks.
    //
    // Returns:
    //     size_t: The total number of free blocks.
    size_t GetNumFreeBlocks() {
        return GetPool().GetNumFreeBlocks();
    }

    // Get the total number of bytes used.
    //
    // Returns:
    //     size_t: The total number of bytes used.
    size_t GetUsedBytes() {
        return GetPool().GetUsedBytes();
    }

    // Get the total number of bytes acquired but not used in the pool.
    //
    // Returns:
    //     size_t: The total number of bytes acquired but not used in the pool.
    size_t GetFreeBytes() {
        return GetPool().GetFreeBytes();
    }

    // Get the total number of bytes acquired in the pool.
    //
    // Returns:
    //     size_t: The total number of bytes acquired in the pool.
    size_t GetTotalBytes() {
        return GetPool().GetTotalBytes();
    }
};

} // namespace internal
} // namespace cumo

#endif /* ifndef CUMO_CUDA_MEMORY_POOL_IMPL_H */
