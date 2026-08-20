#include "memory_pool_impl.hpp"

#include <cstdio>

#include <ruby.h>

namespace cumo {
namespace internal {

void CheckStatus(cudaError_t status) {
    if (status != 0) {
        throw CUDARuntimeError(status);
    }
}

Memory::Memory(size_t size) : size_(size) {
    if (size_ > 0) {
        CheckStatus(cudaGetDevice(&device_id_));
        CheckStatus(cudaMallocManaged(&ptr_, size_, cudaMemAttachGlobal));
        // std::cout << "cudaMalloc " << ptr_ << std::endl;
    }
}

Memory::~Memory() {
    if (size_ > 0) {
        // std::cout << "cudaFree  " << ptr_ << std::endl;
        cudaError_t status = cudaFree(ptr_);
        // CUDA driver may shut down before freeing memory inside memory pool.
        // It is okay to simply ignore because CUDA driver automatically frees memory.
        if (status == cudaSuccess || status == cudaErrorCudartUnloading) {
            return;
        }
        // A destructor is implicitly noexcept, so throwing here would call
        // std::terminate and abort the process. Report the failure instead:
        // cudaFree only fails once the context is already unusable, and the
        // next runtime call reports the same status to the caller anyway.
        std::fprintf(stderr, "cumo: failed to free %zu bytes of device memory: %s\n",
                     size_, cudaGetErrorString(status));
    }
}

std::shared_ptr<Chunk> Split(std::shared_ptr<Chunk>& self, size_t size) {
    assert(self->size_ >= size);
    if (self->size_ == size) {
        return nullptr;
    }

    auto remaining = std::make_shared<Chunk>(self->mem_, self->offset_ + size, self->size_ - size, self->stream_ptr_);
    self->size_ = size;

    if (self->next_) {
        remaining->set_next(std::move(self->next_));
        remaining->next()->set_prev(remaining);
    }
    self->next_ = remaining;
    remaining->set_prev(self);

    return remaining;
}


void Merge(std::shared_ptr<Chunk>& self, std::shared_ptr<Chunk> remaining) {
    assert(remaining != nullptr);
    assert(self->stream_ptr_ == remaining->stream_ptr());
    self->size_ += remaining->size();
    self->next_ = remaining->next();
    if (remaining->next() != nullptr) {
        self->next_->set_prev(self);
    }
}

void SingleDeviceMemoryPool::AppendToFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr) {
    assert(chunk != nullptr && !chunk->in_use());
    size_t bin_index = GetBinIndex(size);

    std::lock_guard<std::recursive_mutex> lock{mutex_};

    Arena& arena = GetArena(stream_ptr);
    ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
    size_t arena_index = std::lower_bound(arena_index_map.begin(), arena_index_map.end(), bin_index) - arena_index_map.begin();
    size_t length = arena_index_map.size();
    if (arena_index >= length || arena_index_map.at(arena_index) != bin_index) {
        arena_index_map.insert(arena_index_map.begin() + arena_index, bin_index);
        arena.insert(arena.begin() + arena_index, FreeList{});
    }
    FreeList& free_list = arena[arena_index];
    free_list.emplace_back(chunk);
}

bool SingleDeviceMemoryPool::RemoveFromFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr) {
    assert(chunk != nullptr && !chunk->in_use());
    size_t bin_index = GetBinIndex(size);

    std::lock_guard<std::recursive_mutex> lock{mutex_};

    Arena& arena = GetArena(stream_ptr);
    ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
    if (arena_index_map.size() == 0) {
        return false;
    }
    size_t arena_index = std::lower_bound(arena_index_map.begin(), arena_index_map.end(), bin_index) - arena_index_map.begin();
    if (arena_index == arena_index_map.size()) {
        // Bin does not exist for the given chunk size.
        return false;
    }
    if (arena_index_map.at(arena_index) != bin_index) {
        return false;
    }
    assert(arena.size() > arena_index);
    FreeList& free_list = arena[arena_index];
    return EraseFromFreeList(free_list, chunk);
}

intptr_t SingleDeviceMemoryPool::Malloc(size_t size, cudaStream_t stream_ptr) {
    if (size == 0) {
        // A zero-sized chunk would share its address with the chunk it was
        // split from, and aliased addresses break the `in_use_` bookkeeping.
        // cudaMalloc returns a null pointer for a zero-sized request as well.
        return 0;
    }
    if (size > kMaxAllocationSize) {
        // Rounding up would wrap around in size_t and the pool would then hand
        // out a chunk far smaller than requested. Such a request can never be
        // satisfied, so report it as out of memory instead.
        throw OutOfMemoryError(size, GetTotalBytes());
    }
    size = GetRoundedSize(size);
    std::shared_ptr<Chunk> chunk = nullptr;

    {
        std::lock_guard<std::recursive_mutex> lock{mutex_};

        // find best-fit, or a smallest larger allocation
        Arena& arena = GetArena(stream_ptr);
        size_t arena_index = GetArenaIndex(size, stream_ptr);
        size_t arena_length = arena.size();
        for (size_t i = arena_index; i < arena_length; ++i) {
            FreeList& free_list = arena[i];
            if (free_list.empty()) {
                continue;
            }
            chunk = PopFromFreeList(free_list);
            if (free_list.empty()) {
                // An emptied bin stays in the arena otherwise, and every later
                // search walks it. Dropping it here invalidates arena and
                // free_list, so nothing below this loop may touch them.
                CompactIndex(stream_ptr, false);
            }
            break;
        }

        // Splitting rewrites the prev/next pointers of the neighbouring
        // chunks, which other threads reach through the free lists, so it has
        // to happen under the same lock as the search above.
        if (chunk != nullptr) {
            std::shared_ptr<Chunk> remaining = Split(chunk, size);
            if (remaining != nullptr) {
                AppendToFreeList(remaining->size(), remaining, stream_ptr);
            }
        }
    }

    if (chunk == nullptr) {
        // cudaMalloc if a cache is not found. This stays outside the lock: it
        // is slow, and a chunk of a fresh allocation has no neighbours for
        // another thread to reach it through.
        std::shared_ptr<Memory> mem = nullptr;
        try {
            mem = std::make_shared<Memory>(size);
        } catch (const CUDARuntimeError& e) {
            if (e.status() != cudaErrorMemoryAllocation) {
                throw;
            }
            // Retry after free all free blocks.
            // NOTE: Anotehr retry after GC is done at cumo_cuda_runtime_malloc.
            FreeAllBlocks();
            try {
                mem = std::make_shared<Memory>(size);
            } catch (const CUDARuntimeError& e) {
                if (e.status() != cudaErrorMemoryAllocation) {
                    throw;
                }
                size_t total = size + GetTotalBytes();
                throw OutOfMemoryError(size, total);
            }
        }
        chunk = std::make_shared<Chunk>(mem, 0, size, stream_ptr);
    }

    assert(chunk != nullptr);
    assert(chunk->stream_ptr() == stream_ptr);
    {
        std::lock_guard<std::recursive_mutex> lock{mutex_};

        chunk->set_in_use(true);
        in_use_.emplace(chunk->ptr(), chunk);
    }
    return chunk->ptr();
}

bool SingleDeviceMemoryPool::Free(intptr_t ptr) {
    std::shared_ptr<Chunk> chunk = nullptr;

    // The whole body runs under the lock. Walking prev/next and merging is a
    // read-modify-write of the chunk graph, which is shared: a neighbour can
    // be split by a concurrent Malloc while this reads its prev/next.
    //
    // The lost merge is the part that shows: two threads freeing neighbouring
    // chunks each look for the other in the free list before the other has
    // appended itself, so both merges fail and the two chunks stay split for
    // good. Fragmentation accumulates from there.
    //
    // mutex_ is recursive, so the RemoveFromFreeList/AppendToFreeList calls
    // below can take it again.
    std::lock_guard<std::recursive_mutex> lock{mutex_};

    {
        // find rather than operator[], which would insert an empty entry for
        // every pointer this pool does not own.
        auto it = in_use_.find(ptr);
        if (it == in_use_.end()) {
            return false;
        }
        chunk = it->second;
        in_use_.erase(it);
        if (!chunk) {
            return false;
        }
        chunk->set_in_use(false);
    }

    // A chunk belongs to the arena of the stream it was allocated on. That is
    // what makes reuse safe: the next allocation from that arena is queued on
    // the same stream as the work that last used the chunk, so the two are
    // ordered. Returning it to another stream's arena hands it to work with no
    // such ordering. Malloc and Merge each state the invariant in an assert,
    // but ruby.h defines NDEBUG, so neither ever runs.
    const cudaStream_t stream_ptr = chunk->stream_ptr();

    if (chunk->next() != nullptr && !chunk->next()->in_use()) {
        if (RemoveFromFreeList(chunk->next()->size(), chunk->next(), stream_ptr)) {
            Merge(chunk, chunk->next());
        }
    }
    if (chunk->prev() != nullptr && !chunk->prev()->in_use()) {
        if (RemoveFromFreeList(chunk->prev()->size(), chunk->prev(), stream_ptr)) {
            chunk = chunk->prev();
            Merge(chunk, chunk->next());
        }
    }
    AppendToFreeList(chunk->size(), chunk, stream_ptr);
    return true;
}

void SingleDeviceMemoryPool::CompactIndex(cudaStream_t stream_ptr, bool free) {
    // need lock ouside this function
    if (!HasArena(stream_ptr)) return;

    Arena new_arena;
    ArenaIndexMap new_arena_index_map;
    Arena& arena = GetArena(stream_ptr);
    ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
    size_t arena_length = arena.size();
    for (size_t arena_index = 0; arena_index < arena_length; ++arena_index) {
        FreeList& free_list = arena[arena_index];
        if (free_list.empty()) {
            continue;
        }
        if (free) {
            FreeList keep_list;
            for (auto chunk : free_list) {
                if (chunk->prev() != nullptr || chunk->next() != nullptr) {
                    keep_list.emplace_back(chunk);
                }
            }
            if (keep_list.size() == 0) {
                continue;
            }
            new_arena_index_map.emplace_back(arena_index_map[arena_index]);
            new_arena.emplace_back(keep_list);
        } else {
            new_arena_index_map.emplace_back(arena_index_map[arena_index]);
            new_arena.emplace_back(free_list);
        }
    }
    if (new_arena.empty()) {
        index_.erase(stream_ptr);
        free_.erase(stream_ptr);
    } else {
        arena_index_map.swap(new_arena_index_map);
        arena.swap(new_arena);
    }
}

// Free all **non-split** chunks in all arenas
void SingleDeviceMemoryPool::FreeAllBlocks() {
    std::lock_guard<std::recursive_mutex> lock{mutex_};

    std::vector<cudaStream_t> keys(free_.size());
    transform(free_.begin(), free_.end(), keys.begin(), [](auto pair) { return pair.first; });
    for (cudaStream_t stream_ptr : keys) {
        CompactIndex(stream_ptr, true);
    }
}

// Free all **non-split** chunks in specified arena
void SingleDeviceMemoryPool::FreeAllBlocks(cudaStream_t stream_ptr) {
    std::lock_guard<std::recursive_mutex> lock{mutex_};

    CompactIndex(stream_ptr, true);
}

size_t SingleDeviceMemoryPool::GetNumFreeBlocks() {
    size_t n = 0;

    std::lock_guard<std::recursive_mutex> lock{mutex_};

    for (auto kv : free_) {
        Arena& arena = kv.second;
        for (auto free_list : arena) {
            n += free_list.size();
        }
    }
    return n;
}

size_t SingleDeviceMemoryPool::GetUsedBytes() {
    size_t size = 0;

    std::lock_guard<std::recursive_mutex> lock{mutex_};

    for (auto kv : in_use_) {
        std::shared_ptr<Chunk>& chunk = kv.second;
        if (chunk) size += chunk->size();
    }
    return size;
}

size_t SingleDeviceMemoryPool::GetFreeBytes() {
    size_t size = 0;

    std::lock_guard<std::recursive_mutex> lock{mutex_};

    for (auto kv : free_) {
        Arena& arena = kv.second;
        for (auto free_list : arena) {
            for (auto chunk : free_list) {
                if (chunk) size += chunk->size();
            }
        }
    }
    return size;
}

} // namespace internal
} // namespace cumo
