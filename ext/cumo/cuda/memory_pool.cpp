#include "cumo/cuda/memory_pool.hpp"

#ifdef NO_RUBY
#else
#include <ruby.h>
#endif

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
        if (status != cudaErrorCudartUnloading) {
            CheckStatus(status);
        }
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
    int bin_index = GetBinIndex(size);
    //rlock.lock_fastrlock(self._free_lock, -1, True)
    //try:
    Arena& arena = GetArena(stream_ptr);
    ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
    int arena_index = std::lower_bound(arena_index_map.begin(), arena_index_map.end(), bin_index) - arena_index_map.begin();
    int length = static_cast<int>(arena_index_map.size());
    if (arena_index >= length || arena_index_map.at(arena_index) != bin_index) {
        arena_index_map.insert(arena_index_map.begin() + arena_index, bin_index);
        arena.insert(arena.begin() + arena_index, FreeList{});
    }
    FreeList& free_list = arena[arena_index];
    free_list.emplace_back(chunk);
    //finally:
    //    rlock.unlock_fastrlock(self._free_lock)
}

bool SingleDeviceMemoryPool::RemoveFromFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr) {
    assert(chunk != nullptr && !chunk->in_use());
    int bin_index = GetBinIndex(size);
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    Arena& arena = GetArena(stream_ptr);
    ArenaIndexMap& arena_index_map = GetArenaIndexMap(stream_ptr);
    if (arena_index_map.size() == 0) {
        return false;
    }
    int arena_index = std::lower_bound(arena_index_map.begin(), arena_index_map.end(), bin_index) - arena_index_map.begin();
    if (arena_index_map.at(arena_index) != bin_index) {
        return false;
    }
    assert(arena.size() > static_cast<size_t>(arena_index));
    FreeList& free_list = arena[arena_index];
    return EraseFromFreeList(free_list, chunk);
    // finally:
    //     rlock.unlock_fastrlock(self._free_lock)
}

intptr_t SingleDeviceMemoryPool::Malloc(size_t size, cudaStream_t stream_ptr) {
    size = GetRoundedSize(size);
    std::shared_ptr<Chunk> chunk = nullptr;

    // find best-fit, or a smallest larger allocation
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    Arena& arena = GetArena(stream_ptr);
    int arena_index = GetArenaIndex(size);
    int arena_length = static_cast<int>(arena.size());
    for (int i = arena_index; i < arena_length; ++i) {
        FreeList& free_list = arena[i];
        if (free_list.empty()) {
            continue;
        }
        chunk = PopFromFreeList(free_list);
        // TODO(sonots): compact_index
        break;
    }
    // finally:
    //     rlock.unlock_fastrlock(self._free_lock)

    if (chunk != nullptr) {
        std::shared_ptr<Chunk> remaining = Split(chunk, size);
        if (remaining != nullptr) {
            AppendToFreeList(remaining->size(), remaining, stream_ptr);
        }
    } else {
        // cudaMalloc if a cache is not found
        std::shared_ptr<Memory> mem = nullptr;
        try {
            mem = std::make_shared<Memory>(size);
        } catch (const CUDARuntimeError& e) {
            if (e.status() != cudaErrorMemoryAllocation) {
                throw;
            }
            FreeAllBlocks();
            try {
                mem = std::make_shared<Memory>(size);
            } catch (const CUDARuntimeError& e) {
                if (e.status() != cudaErrorMemoryAllocation) {
                    throw;
                }
#ifdef NO_RUBY
                size_t total = size + GetTotalBytes();
                throw OutOfMemoryError(size, total);
#else
                rb_funcall(rb_intern("GC"), rb_intern("start"), 0);
                try {
                    mem = std::make_shared<Memory>(size);
                } catch (const CUDARuntimeError& e) {
                    if (e.status() != cudaErrorMemoryAllocation) {
                        throw;
                    }
                    size_t total = size + GetTotalBytes();
                    throw OutOfMemoryError(size, total);
                }
#endif
            }
        }
        chunk = std::make_shared<Chunk>(mem, 0, size, stream_ptr);
    }

    assert(chunk != nullptr);
    assert(chunk->stream_ptr() == stream_ptr);
    //rlock.lock_fastrlock(self._in_use_lock, -1, True)
    //try:
    chunk->set_in_use(true);
    in_use_.emplace(chunk->ptr(), chunk);
    //finally:
    //    rlock.unlock_fastrlock(self._in_use_lock)
    return chunk->ptr();
}

void SingleDeviceMemoryPool::Free(intptr_t ptr, cudaStream_t stream_ptr) {
    //rlock.lock_fastrlock(self._in_use_lock, -1, True)
    //try:
    std::shared_ptr<Chunk> chunk = in_use_[ptr];
    assert(chunk != nullptr);
    chunk->set_in_use(false);
    in_use_.erase(ptr);
    //finally:
    //    rlock.unlock_fastrlock(self._in_use_lock)

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
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    std::vector<cudaStream_t> keys(free_.size());
    transform(free_.begin(), free_.end(), keys.begin(), [](auto pair) { return pair.first; });
    for (cudaStream_t stream_ptr : keys) {
        CompactIndex(stream_ptr, true);
    }
    //finally:
    //    rlock.unlock_fastrlock(self._free_lock)
}

// Free all **non-split** chunks in specified arena
void SingleDeviceMemoryPool::FreeAllBlocks(cudaStream_t stream_ptr) {
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    CompactIndex(stream_ptr, true);
    //finally:
    //    rlock.unlock_fastrlock(self._free_lock)
}

size_t SingleDeviceMemoryPool::GetNumFreeBlocks() {
    size_t n = 0;
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    for (auto kv : free_) {
        Arena& arena = kv.second;
        for (auto free_list : arena) {
            n += free_list.size();
        }
    }
    // finally:
    //     rlock.unlock_fastrlock(self._free_lock)
    return n;
}

size_t SingleDeviceMemoryPool::GetUsedBytes() {
    size_t size = 0;
    // rlock.lock_fastrlock(self._in_use_lock, -1, True)
    // try:
    for (auto kv : in_use_) {
        std::shared_ptr<Chunk>& chunk = kv.second;
        size += chunk->size();
    }
    // finally:
    //     rlock.unlock_fastrlock(self._in_use_lock)
    return size;
}

size_t SingleDeviceMemoryPool::GetFreeBytes() {
    size_t size = 0;
    // rlock.lock_fastrlock(self._free_lock, -1, True)
    // try:
    for (auto kv : free_) {
        Arena& arena = kv.second;
        for (auto free_list : arena) {
            for (auto chunk : free_list) {
                size += chunk->size();
            }
        }
    }
    // finally:
    //     rlock.unlock_fastrlock(self._free_lock)
    return size;
}

} // namespace internal
} // namespace cumo
