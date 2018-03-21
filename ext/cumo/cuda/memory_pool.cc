#include "cumo/cuda/memory_pool.h"

namespace cumo {
namespace internal {

void CheckStatus(cudaError_t status) {
    if (status != 0) {
        throw CUDARuntimeError(status);
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

void MemoryPool::AppendToFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr) {
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

bool MemoryPool::RemoveFromFreeList(size_t size, std::shared_ptr<Chunk>& chunk, cudaStream_t stream_ptr) {
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

intptr_t MemoryPool::Malloc(size_t size) {
    // if (size == 0) return 0;
    size = GetRoundedSize(size);
    // TODO: support cuda stream
    // stream_ptr = stream_module.get_current_stream_ptr()
    cudaStream_t stream_ptr = 0;

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
            throw;
            // TODO(sonots): Complete below
            //if (e.status() != cudaErrorMemoryAllocation) {
            //    throw;
            //}
            //free_all_blocks();
            //try {
            //    mem = std::make_shared<Memory>(size);
            //} catch (const CUDARuntimeError& e) {  
            //    if (e.status() != cudaErrorMemoryAllocation) {
            //        throw;
            //    }
            //    // GC.start
            //    try {
            //        mem = std::make_shared<Memory>(size);
            //    } catch (const CUDARuntimeError& e) {  
            //        if (e.status() != cudaErrorMemoryAllocation) {
            //            throw;
            //        } else {
            //            size_t total = size + total_bytes();
            //            throw OutOfMemoryError(size, total);
            //        }
            //    }
            //}
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

void MemoryPool::Free(intptr_t ptr) {
    //rlock.lock_fastrlock(self._in_use_lock, -1, True)
    //try:
    std::shared_ptr<Chunk> chunk = in_use_[ptr];
    if (chunk == nullptr) {
        throw std::runtime_error("Cannot free out-of-pool memory");
    }
    chunk->set_in_use(false);
    in_use_.erase(ptr);
    //finally:
    //    rlock.unlock_fastrlock(self._in_use_lock)

    //TODO(sonots): Support stream
    cudaStream_t stream_ptr = chunk->stream_ptr();

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

} // namespace internal
} // namespace cumo
