#include <ruby.h>
#include <cuda_runtime.h>
#include "cumo/cuda/memory.h"
#include "cumo/cuda/runtime.h"

#include <cassert>
#include <memory>
#include <unordered_map>
#include <vector>
#include <iostream>

// cudaMalloc() is aligned to at least 512 bytes
// cf. https://gist.github.com/sonots/41daaa6432b1c8b27ef782cd14064269
const int kRoundSize = 512; // bytes

namespace {

// Memory allocation on a CUDA device.
//
// This class provides an RAII interface of the CUDA memory allocation.
class Memory {
public:
    // size: Size of the memory allocation in bytes.
    Memory(size_t size) : size_(size) {
        if (size_ > 0) {
            cumo_cuda_runtime_check_status(cudaGetDevice(&device_id_));
            cumo_cuda_runtime_check_status(cudaMallocManaged(&ptr_, size_, cudaMemAttachGlobal));
        }
    }

    ~Memory() {
        if (size_ > 0) {
            cumo_cuda_runtime_check_status(cudaFree(ptr_));
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


// MEMO: memory pool holds chunk, and chunk holds a memory (RAII)

// A chunk points to a device memory.
//
// A chunk might be a splitted memory block from a larger allocation.
// The prev/next pointers contruct a doubly-linked list of memory addresses
// sorted by base address that must be contiguous.
//
// Attributes:
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

    // Split contiguous block of a larger allocation
    std::shared_ptr<Chunk> split(size_t size) {
        assert(size_ >= size);
        if (size_ == size) {
            return nullptr;
        }

        auto remaining = std::make_shared<Chunk>(mem_, offset_ + size, size_ - size, stream_ptr_);
        size_ = size;

        if (next_) {
            remaining->set_next(std::move(next_));
            remaining->next()->set_prev(remaining);
        }
        next_ = remaining;
        remaining->set_prev(shared_from_this());

        return remaining;
    }

    // Merge previously splitted block (chunk)
    void merge(std::shared_ptr<Chunk>& remaining) {
        assert(stream_ptr_ == remaining->stream_ptr());
        size_ += remaining->size();
        next_ = remaining->next();
        if (remaining->next()) {
            next_->set_prev(shared_from_this());
        }
    }
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
};

class Pool {
    public:
    void free(intptr_t ptr, size_t size) {}
};

// Memory allocation for a memory pool.
//
// The instance of this class is created by memory pool allocator, so user
// should not instantiate it by hand.
class PooledMemory {
public:
    PooledMemory(const std::shared_ptr<Chunk>& chunk, Pool& pool) :
        ptr_(chunk->ptr()), size_(chunk->size()), device_id_(chunk->device_id()), pool_(pool) {}

    // Returns the pointer value to the head of the allocation.
    intptr_t ptr() const { return ptr_; }
    size_t size() const { return size_; }
    int device_id() const { return device_id_; }

    // TODO: call free via GC

    // Frees the memory buffer and returns it to the memory pool.
    //
    // This function actually does not free the buffer. It just returns the
    // buffer to the memory pool for reuse.
    void free() {
        if (ptr_ == 0) return;
        intptr_t ptr = ptr_;
        size_t size = size_;
        ptr_ = 0;
        size_ = 0;
        pool_.free(ptr, size);
    }

private:
    // Pointer to the place within the buffer.
    intptr_t ptr_ = 0;
    // Size of the memory allocation in bytes.
    size_t size_ = 0;
    // GPU device id whose memory the pointer refers to.
    int device_id_ = -1;
    // Memory pool instance
    Pool& pool_;
};

/*
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
    std::unordered_map<void*, Chunk> in_use_;  // {ptr: Chunk}
    std::unordered_map<intptr_t, std::vector<std::unordered_set<Chunk>>> free_; // {stream_ptr: {index: set(Chunk)}}
    std::unordered_map<intptr_t, std::vector<int>> index_; // {stream_ptr: [index]}

public:
    SingleDeviceMemoryPool() {
        cumo_cuda_runtime_check_status(cudaGetDevice(&device_id));
    }

    // Round up the memory size to fit memory alignment of cudaMalloc.
    size_t GetRoundedSize(size_t size) {
        return ((size + kRoundSize - 1) / kRoundSize) * kRoundSize;
    }

    // Get appropriate bins index from the memory size
    int GetBinIndexFromSize(size_t size) {
        return (size - 1) / kRoundSize;
    }

    // Get appropriate arena (list of bins) of a given stream
    std::vector<std::unordered_set<Chunk>>& GetArena(size_t stream_ptr) {
        return free_[stream_ptr];  // initialize if not exist
    }

    // Get appropriate arena sparse index of a given stream
    std::vector<int>& GetArenaIndex(size_t stream_ptr) {
        return index_[stream_ptr];
    }

    // TODO
    void AppendToFreeList(size_t size, Chunk chunk, size_t stream_ptr) {
        cdef int index, bin_index
        cdef list arena
        cdef set free_list
        cdef vector.vector[int]* arena_index

        int bin_index = GetBinIndexFromSize(size);
        //rlock.lock_fastrlock(self._free_lock, -1, True)
        //try:
        arena = GetArean(stream_ptr);
        std::vector<int>& arena_index = GetArenaIndex(stream_ptr);
        int index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin();
        int size = static_cast<int>(arena_index.size());
        if (index < size and arena_index.at(index) == bin_index) {
            free_list = arena[index];
            if (free_list is nullptr) {
                arena[index] = free_list = set();
            }
        } else {
            free_list = set();
            arena_index.insert(arena_index.begin() + index, bin_index);
            arena.insert(index, free_list);
        }
        free_list.add(chunk);
        //finally:
        //    rlock.unlock_fastrlock(self._free_lock)
    }

    bool RemoveFromFreeList(size_t size, Chunk chunk, size_t stream_ptr) {
        cdef int index, bin_index
        cdef list arena
        cdef set free_list

        bin_index = GetBinIndexFromSize(size);
        // rlock.lock_fastrlock(self._free_lock, -1, True)
        // try:
        arena = GetArena(stream_ptr);
        std::vector<int>& arena_index = GetArenaIndex(stream_ptr);
        if (arena_index.size() == 0) {
            return false;
        }
        index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin();
        if (arena_index.at(index) != bin_index) {
            return false;
        }
        free_list = arena[index];
        if (free_list and chunk in free_list) {
            free_list.remove(chunk);
            if (len(free_list) == 0) {
                arena[index] = nullptr;
            }
            return true;
        }
        // finally:
        //     rlock.unlock_fastrlock(self._free_lock)
        return false;
    }

    MemoryPointer Alloc(size_t rounded_size) {
        return self._allocator(rounded_size);
    }

    MemoryPointer Malloc(size_t size) {
        rounded_size = self.GetRoundedSize(size);
        return self._malloc(rounded_size);
    }

    MemoryPointer Malloc(size_t size) {
        cdef set free_list
        cdef _Chunk chunk = None
        cdef _Chunk remaining
        cdef int bin_index, index, length

        if size == 0:
            return MemoryPointer(Memory(0), 0)

        stream_ptr = stream_module.get_current_stream_ptr()

        bin_index = self.GetBinIndexFromSize(size)
        # find best-fit, or a smallest larger allocation
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            arena = self._arena(stream_ptr)
            arena_index = self.GetArenaIndex(stream_ptr)
            index = algorithm.lower_bound(
                arena_index.begin(), arena_index.end(),
                bin_index) - arena_index.begin()
            length = arena_index.size()
            for i in range(index, length):
                free_list = arena[i]
                if free_list is None:
                    continue
                assert len(free_list) > 0
                chunk = free_list.pop()
                if len(free_list) == 0:
                    arena[i] = None
                if i - index >= _index_compaction_threshold:
                    _compact_index(self, stream_ptr, False)
                break
        finally:
            rlock.unlock_fastrlock(self._free_lock)

        if chunk is not None:
            remaining = chunk.split(size)
            if remaining is not None:
                self.AppendToFreeList(remaining.size, remaining,
                                          stream_ptr)
        else:
            # cudaMalloc if a cache is not found
            try:
                mem = self._alloc(size).mem
            except runtime.CUDARuntimeError as e:
                if e.status != runtime.errorMemoryAllocation:
                    raise
                self.free_all_blocks()
                try:
                    mem = self._alloc(size).mem
                except runtime.CUDARuntimeError as e:
                    if e.status != runtime.errorMemoryAllocation:
                        raise
                    gc.collect()
                    try:
                        mem = self._alloc(size).mem
                    except runtime.CUDARuntimeError as e:
                        if e.status != runtime.errorMemoryAllocation:
                            raise
                        else:
                            total = size + self.total_bytes()
                            raise OutOfMemoryError(size, total)
            chunk = _Chunk.__new__(_Chunk)
            chunk._init(mem, 0, size, stream_ptr)

        assert chunk.stream_ptr == stream_ptr
        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            self._in_use[chunk.ptr] = chunk
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        pmem = PooledMemory(chunk, self._weakref)
        return MemoryPointer(pmem, 0)

    cpdef free(self, size_t ptr, Py_ssize_t size):
        cdef set free_list
        cdef _Chunk chunk

        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            chunk = self._in_use.pop(ptr)
        except KeyError:
            raise RuntimeError('Cannot free out-of-pool memory')
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        stream_ptr = chunk.stream_ptr

        if chunk.next is not None:
            if self.RemoveFromFreeList(chunk.next.size, chunk.next,
                                           stream_ptr):
                chunk.merge(chunk.next)

        if chunk.prev is not None:
            if self.RemoveFromFreeList(chunk.prev.size, chunk.prev,
                                           stream_ptr):
                chunk = chunk.prev
                chunk.merge(chunk.next)

        self.AppendToFreeList(chunk.size, chunk, stream_ptr)

    cpdef free_all_blocks(self, stream=None):
        """Free all **non-split** chunks"""
        cdef size_t stream_ptr

        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            # free blocks in all arenas
            if stream is None:
                for stream_ptr in list(self._free.iterkeys()):
                    _compact_index(self, stream_ptr, True)
            else:
                _compact_index(self, stream.ptr, True)
        finally:
            rlock.unlock_fastrlock(self._free_lock)

    cpdef free_all_free(self):
        warnings.warn(
            'free_all_free is deprecated. Use free_all_blocks instead.',
            DeprecationWarning)
        self.free_all_blocks()

    cpdef n_free_blocks(self):
        cdef Py_ssize_t n = 0
        cdef set free_list
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._free.itervalues():
                for v in arena:
                    if v is not None:
                        n += len(v)
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return n

    cpdef used_bytes(self):
        cdef Py_ssize_t size = 0
        cdef _Chunk chunk
        rlock.lock_fastrlock(self._in_use_lock, -1, True)
        try:
            for chunk in self._in_use.itervalues():
                size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._in_use_lock)
        return size

    cpdef free_bytes(self):
        cdef Py_ssize_t size = 0
        cdef set free_list
        cdef _Chunk chunk
        rlock.lock_fastrlock(self._free_lock, -1, True)
        try:
            for arena in self._free.itervalues():
                for free_list in arena:
                    if free_list is None:
                        continue
                    for chunk in free_list:
                        size += chunk.size
        finally:
            rlock.unlock_fastrlock(self._free_lock)
        return size

    cpdef total_bytes(self):
        return self.used_bytes() + self.free_bytes()

*/
} // namespace

std::unordered_map<void*, size_t> in_use;
std::vector<std::vector<void*>> free_bins;

inline int GetIndex(size_t size) { return size / kRoundSize; }

char*
cumo_cuda_runtime_malloc(size_t size)
{
    void *ptr;
    int index = GetIndex(size);
    if (index >= static_cast<int>(free_bins.size())) {
        free_bins.resize(index + 1);
    }
    if (free_bins[index].empty()) {
        cudaError_t status = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
        cumo_cuda_runtime_check_status(status);
        //std::cout << "malloc " << (size_t)(ptr) << " " << size << " " << index << std::endl;
        // TODO(sonots): If fails to allocate, once free all memory
        //cudaError_t status = cudaFree((void*)ptr);
        //cumo_cuda_runtime_check_status(status);
    } else {
        // TODO(sonots): atomic
        ptr = free_bins[index].back();
        free_bins[index].pop_back();
        //std::cout << "reuse  " << (size_t)(ptr) << " " << size << " " << index << std::endl;
    }
    in_use.emplace(ptr, size);
    return (char*)ptr;
}

void
cumo_cuda_runtime_free(char *ptr)
{
    size_t size = in_use[ptr];
    int index = GetIndex(size);
    // TODO(sonots): atomic
    //std::cout << "free   " << (size_t)(ptr) << " " << size << " " << index << std::endl;
    free_bins[index].emplace_back(ptr);
    in_use.erase(ptr);
}
