#include "cumo/cuda/memory_pool.hpp"

#include <cassert>
#include <memory>
#include <iostream>

// TODO(sonots): Use googletest?
// TODO(sonots): Provide clean way to build this test outside extconf.rb

namespace cumo {
namespace internal {

class TestChunk {
private:
    cudaStream_t stream_ptr_ = 0;

public:
    TestChunk() {}

    void Run() {
        TestSplit();
        TestMerge();
    }

    void TestSplit() {
        auto mem = std::make_shared<Memory>(kRoundSize * 4);
        auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);

        auto tail = Split(chunk, kRoundSize * 2);
        assert(chunk->ptr() == mem->ptr());
        assert(chunk->offset() == 0);
        assert(chunk->size() == kRoundSize * 2);
        assert(chunk->prev() == nullptr);
        assert(chunk->next()->ptr() == tail->ptr());
        assert(chunk->stream_ptr() == stream_ptr_);
        assert(tail->ptr() == mem->ptr() + kRoundSize * 2);
        assert(tail->offset() == kRoundSize * 2);
        assert(tail->size() == kRoundSize * 2);
        assert(tail->prev()->ptr() == chunk->ptr());
        assert(tail->next() == nullptr);
        assert(tail->stream_ptr() == stream_ptr_);

        auto tail_of_head = Split(chunk, kRoundSize);
        assert(chunk->ptr() == mem->ptr());
        assert(chunk->offset() == 0);
        assert(chunk->size() == kRoundSize);
        assert(chunk->prev() == nullptr);
        assert(chunk->next()->ptr() == tail_of_head->ptr());
        assert(chunk->stream_ptr() == stream_ptr_);
        assert(tail_of_head->ptr() == mem->ptr() + kRoundSize);
        assert(tail_of_head->offset() == kRoundSize);
        assert(tail_of_head->size() == kRoundSize);
        assert(tail_of_head->prev()->ptr() == chunk->ptr());
        assert(tail_of_head->next()->ptr() == tail->ptr());
        assert(tail_of_head->stream_ptr() == stream_ptr_);

        auto tail_of_tail = Split(tail, kRoundSize);
        assert(tail->ptr() == chunk->ptr() + kRoundSize * 2);
        assert(tail->offset() == kRoundSize * 2);
        assert(tail->size() == kRoundSize);
        assert(tail->prev()->ptr() == tail_of_head->ptr());
        assert(tail->next()->ptr() == tail_of_tail->ptr());
        assert(tail->stream_ptr() == stream_ptr_);
        assert(tail_of_tail->ptr() == mem->ptr() + kRoundSize * 3);
        assert(tail_of_tail->offset() == kRoundSize * 3);
        assert(tail_of_tail->size() == kRoundSize);
        assert(tail_of_tail->prev()->ptr() == tail->ptr());
        assert(tail_of_tail->next() == nullptr);
        assert(tail_of_tail->stream_ptr() == stream_ptr_);
    }

    void TestMerge() {
        auto mem = std::make_shared<Memory>(kRoundSize * 4);
        auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);

        auto chunk_ptr = chunk->ptr();
        auto chunk_offset = chunk->offset();
        auto chunk_size = chunk->size();

        auto tail = Split(chunk, kRoundSize * 2);
        auto head = chunk;
        auto head_ptr = head->ptr();
        auto head_offset = head->offset();
        auto head_size = head->size();
        auto tail_ptr = tail->ptr();
        auto tail_offset = tail->offset();
        auto tail_size = tail->size();

        auto tail_of_head = Split(head, kRoundSize);
        auto tail_of_tail = Split(tail, kRoundSize);

        Merge(head, tail_of_head);
        assert(head->ptr() == head_ptr);
        assert(head->offset() == head_offset);
        assert(head->size() == head_size);
        assert(head->prev() == nullptr);
        assert(head->next()->ptr() == tail_ptr);
        assert(head->stream_ptr() == stream_ptr_);

        Merge(tail, tail_of_tail);
        assert(tail->ptr() == tail_ptr);
        assert(tail->offset() == tail_offset);
        assert(tail->size() == tail_size);
        assert(tail->prev()->ptr() == head_ptr);
        assert(tail->next() == nullptr);
        assert(tail->stream_ptr() == stream_ptr_);

        Merge(head, tail);
        assert(head->ptr() == chunk_ptr);
        assert(head->offset() == chunk_offset);
        assert(head->size() == chunk_size);
        assert(head->prev() == nullptr);
        assert(head->next() == nullptr);
        assert(head->stream_ptr() == stream_ptr_);
    }
};

class TestMemoryPool {
private:
    std::shared_ptr<MemoryPool> pool_;
    cudaStream_t stream_ptr_ = 0;

public:
    TestMemoryPool() {}

    void SetUp() {
        pool_ = std::make_shared<MemoryPool>();
    }

    void TearDown() {
        pool_.reset();
    }

    void Run() {
        TearDown(); SetUp(); TestGetRoundedSize();
        TearDown(); SetUp(); TestGetBinIndex();
        TearDown(); SetUp(); TestAppendToFreeList();
        TearDown(); SetUp(); TestRemoveFromFreeList();
        TearDown(); SetUp(); TestMalloc();
        TearDown(); SetUp(); TestFree();
        TearDown(); SetUp(); TestMallocSplit();
        TearDown(); SetUp(); TestFreeMerge();
        TearDown(); SetUp(); TestFreeDifferentSize();
        TearDown(); SetUp(); TestFreeAllBlocks();
        TearDown(); SetUp(); TestFreeAllBlocksSplit();
        TearDown(); SetUp(); TestGetUsedBytes();
        TearDown(); SetUp(); TestGetFreeBytes();
        TearDown(); SetUp(); TestGetTotalBytes();
        TearDown();
    }

    void TestGetRoundedSize() {
        assert(pool_->GetRoundedSize(kRoundSize - 1) == kRoundSize);
        assert(pool_->GetRoundedSize(kRoundSize) == kRoundSize);
        assert(pool_->GetRoundedSize(kRoundSize + 1) == kRoundSize * 2);
    }

    void TestGetBinIndex() {
        assert(pool_->GetBinIndex(kRoundSize - 1) == 0);
        assert(pool_->GetBinIndex(kRoundSize) == 0);
        assert(pool_->GetBinIndex(kRoundSize + 1) == 1);
    }

    void TestAppendToFreeList() {
        Arena& arena = pool_->GetArena(stream_ptr_);
        ArenaIndexMap& arena_index_map = pool_->GetArenaIndexMap(stream_ptr_);

        {
            auto mem = std::make_shared<Memory>(kRoundSize * 4);
            auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);
            pool_->AppendToFreeList(chunk->size(), chunk, stream_ptr_);
        }
        assert(arena.size() == 1);
        assert(arena[0].size() == 1);
        assert(arena_index_map.size() == 1);
        assert(arena_index_map[0] == 3);

        // insert to same arena index
        {
            auto mem = std::make_shared<Memory>(kRoundSize * 4);
            auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);
            pool_->AppendToFreeList(chunk->size(), chunk, stream_ptr_);
        }
        assert(arena.size() == 1);
        assert(arena[0].size() == 2);
        assert(arena_index_map.size() == 1);
        assert(arena_index_map[0] == 3);

        // insert to larger arena index
        {
            auto mem = std::make_shared<Memory>(kRoundSize * 5);
            auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);
            pool_->AppendToFreeList(chunk->size(), chunk, stream_ptr_);
        }
        assert(arena.size() == 2);
        assert(arena[0].size() == 2);
        assert(arena[1].size() == 1);
        assert(arena_index_map.size() == 2);
        assert(arena_index_map[0] == 3);
        assert(arena_index_map[1] == 4);

        // insert to smaller arena index
        {
            auto mem = std::make_shared<Memory>(kRoundSize * 3);
            auto chunk = std::make_shared<Chunk>(mem, 0, mem->size(), stream_ptr_);
            pool_->AppendToFreeList(chunk->size(), chunk, stream_ptr_);
        }
        assert(arena.size() == 3);
        assert(arena[0].size() == 1);
        assert(arena[1].size() == 2);
        assert(arena[2].size() == 1);
        assert(arena_index_map.size() == 3);
        assert(arena_index_map[0] == 2);
        assert(arena_index_map[1] == 3);
        assert(arena_index_map[2] == 4);
    }

    // TODO(sonots): Fix after implementing compaction
    void TestRemoveFromFreeList() {
        Arena& arena = pool_->GetArena(stream_ptr_);
        ArenaIndexMap& arena_index_map = pool_->GetArenaIndexMap(stream_ptr_);

        auto mem1 = std::make_shared<Memory>(kRoundSize * 4);
        auto chunk1 = std::make_shared<Chunk>(mem1, 0, mem1->size(), stream_ptr_);
        pool_->AppendToFreeList(chunk1->size(), chunk1, stream_ptr_);

        auto mem2 = std::make_shared<Memory>(kRoundSize * 4);
        auto chunk2 = std::make_shared<Chunk>(mem2, 0, mem2->size(), stream_ptr_);
        pool_->AppendToFreeList(chunk2->size(), chunk2, stream_ptr_);

        auto mem3 = std::make_shared<Memory>(kRoundSize * 5);
        auto chunk3 = std::make_shared<Chunk>(mem3, 0, mem3->size(), stream_ptr_);
        pool_->AppendToFreeList(chunk3->size(), chunk3, stream_ptr_);

        auto mem4 = std::make_shared<Memory>(kRoundSize * 3);
        auto chunk4 = std::make_shared<Chunk>(mem4, 0, mem4->size(), stream_ptr_);
        pool_->AppendToFreeList(chunk4->size(), chunk4, stream_ptr_);

        // remove one from two
        pool_->RemoveFromFreeList(chunk1->size(), chunk1, stream_ptr_);
        assert(arena.size() == 3);
        assert(arena[0].size() == 1);
        assert(arena[1].size() == 1);
        assert(arena[2].size() == 1);
        assert(arena_index_map.size() == 3);
        assert(arena_index_map[0] == 2);
        assert(arena_index_map[1] == 3);
        assert(arena_index_map[2] == 4);

        // remove two from two
        pool_->RemoveFromFreeList(chunk2->size(), chunk2, stream_ptr_);
        assert(arena.size() == 3);
        assert(arena[0].size() == 1);
        assert(arena[1].size() == 0);
        assert(arena[2].size() == 1);
        assert(arena_index_map.size() == 3);
        assert(arena_index_map[0] == 2);
        assert(arena_index_map[1] == 3);
        assert(arena_index_map[2] == 4);

        pool_->RemoveFromFreeList(chunk3->size(), chunk3, stream_ptr_);
        assert(arena.size() == 3);
        assert(arena[0].size() == 1);
        assert(arena[1].size() == 0);
        assert(arena[2].size() == 0);
        assert(arena_index_map.size() == 3);
        assert(arena_index_map[0] == 2);
        assert(arena_index_map[1] == 3);
        assert(arena_index_map[2] == 4);

        pool_->RemoveFromFreeList(chunk4->size(), chunk4, stream_ptr_);
        assert(arena.size() == 3);
        assert(arena[0].size() == 0);
        assert(arena[1].size() == 0);
        assert(arena[2].size() == 0);
        assert(arena_index_map.size() == 3);
        assert(arena_index_map[0] == 2);
        assert(arena_index_map[1] == 3);
        assert(arena_index_map[2] == 4);
    }

    void TestMalloc() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 4);
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        intptr_t p3 = pool_->Malloc(kRoundSize * 8);
        assert(p1 != p2);
        assert(p1 != p3);
        assert(p2 != p3);
    }

    void TestFree() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p1);
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        assert(p1 == p2);
    }

    void TestMallocSplit() {
        intptr_t p = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p);
        intptr_t head = pool_->Malloc(kRoundSize * 2);
        intptr_t tail = pool_->Malloc(kRoundSize * 2);
        assert(p == head);
        assert(p + kRoundSize * 2 == tail);
    }

    void TestFreeMerge() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p1);

        // merge head into tail
        {
            intptr_t head = pool_->Malloc(kRoundSize * 2);
            intptr_t tail = pool_->Malloc(kRoundSize * 2);
            assert(p1 == head);
            pool_->Free(tail);
            pool_->Free(head);
            intptr_t p2 = pool_->Malloc(kRoundSize * 4);
            assert(p1 == p2);
            pool_->Free(p2);
        }

        // merge tail into head
        {
            intptr_t head = pool_->Malloc(kRoundSize * 2);
            intptr_t tail = pool_->Malloc(kRoundSize * 2);
            assert(p1 == head);
            pool_->Free(head);
            pool_->Free(tail);
            intptr_t p2 = pool_->Malloc(kRoundSize * 4);
            assert(p1 == p2);
            pool_->Free(p2);
        }
    }

    void TestFreeDifferentSize() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p1);
        intptr_t p2 = pool_->Malloc(kRoundSize * 8);
        assert(p1 != p2);
    }

    void TestFreeAllBlocks() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p1);
        pool_->FreeAllBlocks();
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        // assert(p1 != p2); // cudaMalloc gets same address ...
        pool_->Free(p2);
    }

    void TestFreeAllBlocksSplit() {
        // do not free splitted blocks
        intptr_t p = pool_->Malloc(kRoundSize * 4);
        pool_->Free(p);
        intptr_t head = pool_->Malloc(kRoundSize * 2);
        intptr_t tail = pool_->Malloc(kRoundSize * 2);
        pool_->Free(tail);
        pool_->FreeAllBlocks();
        intptr_t p2 = pool_->Malloc(kRoundSize * 2);
        assert(tail == p2);
        pool_->Free(head);
    }

    // void TestFreeAllBlocksStream() {
    //      intptr_t p1 = pool_->Malloc(kRoundSize * 4);
    //      pool_->Free(p1);
    //      with self.stream:
    //          p2 = pool_->Malloc(kRoundSize * 4)
    //          ptr2 = p2.ptr
    //          del p2
    //      pool_->free_all_blocks(stream=stream_module.Stream.null)
    //      p3 = pool_->Malloc(kRoundSize * 4)
    //      self.assertNotEqual(ptr1, p3.ptr)
    //      self.assertNotEqual(ptr2, p3.ptr)
    //      with self.stream:
    //          p4 = pool_->Malloc(kRoundSize * 4)
    //          self.assertNotEqual(ptr1, p4.ptr)
    //          assert(ptr2, p4.ptr)

    // def test_free_all_blocks_all_streams(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     ptr1 = p1.ptr
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 4)
    //         ptr2 = p2.ptr
    //         del p2
    //     pool_.free_all_blocks()
    //     p3 = pool_.Malloc(kRoundSize * 4)
    //     self.assertNotEqual(ptr1, p3.ptr)
    //     self.assertNotEqual(ptr2, p3.ptr)
    //     with self.stream:
    //         p4 = pool_.Malloc(kRoundSize * 4)
    //         self.assertNotEqual(ptr1, p4.ptr)
    //         self.assertNotEqual(ptr2, p4.ptr)

    void TestGetUsedBytes() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 2);
        assert(kRoundSize * 2 == pool_->GetUsedBytes());
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        assert(kRoundSize * 6 == pool_->GetUsedBytes());
        pool_->Free(p2);
        assert(kRoundSize * 2 == pool_->GetUsedBytes());
        pool_->Free(p1);
        assert(kRoundSize * 0 == pool_->GetUsedBytes());
        intptr_t p3 = pool_->Malloc(kRoundSize * 1);
        assert(kRoundSize * 1 == pool_->GetUsedBytes());
        pool_->Free(p3);
    }

    // def test_used_bytes_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 2, pool_.used_bytes())
    //     del p2

    void TestGetFreeBytes() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 2);
        assert(kRoundSize * 0 == pool_->GetFreeBytes());
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        assert(kRoundSize * 0 == pool_->GetFreeBytes());
        pool_->Free(p2);
        assert(kRoundSize * 4 == pool_->GetFreeBytes());
        pool_->Free(p1);
        assert(kRoundSize * 6 == pool_->GetFreeBytes());
        intptr_t p3 = pool_->Malloc(kRoundSize * 1);
        assert(kRoundSize * 5 == pool_->GetFreeBytes());
        pool_->Free(p3);
    }

    // def test_free_bytes_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 4, pool_.free_bytes())
    //     del p2

    void TestGetTotalBytes() {
        intptr_t p1 = pool_->Malloc(kRoundSize * 2);
        assert(kRoundSize * 2 == pool_->GetTotalBytes());
        intptr_t p2 = pool_->Malloc(kRoundSize * 4);
        assert(kRoundSize * 6 == pool_->GetTotalBytes());
        pool_->Free(p1);
        assert(kRoundSize * 6 == pool_->GetTotalBytes());
        pool_->Free(p2);
        assert(kRoundSize * 6 == pool_->GetTotalBytes());
        intptr_t p3 = pool_->Malloc(kRoundSize * 1);
        assert(kRoundSize * 6 == pool_->GetTotalBytes());
        pool_->Free(p3);
    }

    // def test_total_bytes_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 6, pool_.total_bytes())
    //     del p2

};

}  // namespace internal
}  // namespace cumo

int main() {
    cumo::internal::TestChunk{}.Run();
    cumo::internal::TestMemoryPool{}.Run();
    return 0;
}
