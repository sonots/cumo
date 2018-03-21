#include "cumo/cuda/memory_pool.h"

#include <cassert>
#include <memory>

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

        auto tail = chunk->Split(kRoundSize * 2);
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

        auto tail_of_head = chunk->Split(kRoundSize);
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

        auto tail_of_tail = tail->Split(kRoundSize);
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

        auto tail = chunk->Split(kRoundSize * 2);
        auto head = chunk;
        auto head_ptr = head->ptr();
        auto head_offset = head->offset();
        auto head_size = head->size();
        auto tail_ptr = tail->ptr();
        auto tail_offset = tail->offset();
        auto tail_size = tail->size();

        auto tail_of_head = head->Split(kRoundSize);
        auto tail_of_tail = tail->Split(kRoundSize);

        head->Merge(tail_of_head);
        assert(head->ptr() == head_ptr);
        assert(head->offset() == head_offset);
        assert(head->size() == head_size);
        assert(head->prev() == nullptr);
        assert(head->next()->ptr() == tail_ptr);
        assert(head->stream_ptr() == stream_ptr_);

        tail->Merge(tail_of_tail);
        assert(tail->ptr() == tail_ptr);
        assert(tail->offset() == tail_offset);
        assert(tail->size() == tail_size);
        assert(tail->prev()->ptr() == head_ptr);
        assert(tail->next() == nullptr);
        assert(tail->stream_ptr() == stream_ptr_);

        head->Merge(tail);
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
    MemoryPool pool_;
    cudaStream_t stream_ptr_ = 0;

public:
    TestMemoryPool() {}

    void SetUp() {
        pool_ = MemoryPool();
    }

    void TestGetRoundedSize() {
        assert(pool_.GetRoundedSize(kRoundSize - 1) == kRoundSize);
        assert(pool_.GetRoundedSize(kRoundSize) == kRoundSize);
        assert(pool_.GetRoundedSize(kRoundSize + 1) == kRoundSize * 2);
    }

    void TestGetBinIndex() {
        assert(pool_.GetBinIndex(kRoundSize - 1) == 0);
        assert(pool_.GetBinIndex(kRoundSize) == 0);
        assert(pool_.GetBinIndex(kRoundSize + 1) == 1);
    }

    void TestMalloc() {
        intptr_t p1 = pool_.Malloc(kRoundSize * 4);
        intptr_t p2 = pool_.Malloc(kRoundSize * 4);
        intptr_t p3 = pool_.Malloc(kRoundSize * 8);
        assert(p1 != p2);
        assert(p1 != p3);
        assert(p2 != p3);
    }

    void TestFree() {
        intptr_t p1 = pool_.Malloc(kRoundSize * 4);
        pool_.Free(p1);
        intptr_t p2 = pool_.Malloc(kRoundSize * 4);
        assert(p1 == p2);
    }

    void TestMallocSplit() {
        intptr_t p = pool_.Malloc(kRoundSize * 4);
        pool_.Free(p);
        intptr_t head = pool_.Malloc(kRoundSize * 2);
        intptr_t tail = pool_.Malloc(kRoundSize * 2);
        assert(p == head);
        assert(p + kRoundSize * 2 == tail);
    }

    void TestFreeMerge() {
        intptr_t p1 = pool_.Malloc(kRoundSize * 4);
        pool_.Free(p1);

        // merge head into tail
        {
            intptr_t head = pool_.Malloc(kRoundSize * 2);
            intptr_t tail = pool_.Malloc(kRoundSize * 2);
            assert(p1 == head);
            pool_.Free(tail);
            pool_.Free(head);
            intptr_t p2 = pool_.Malloc(kRoundSize * 4);
            assert(p1 == p2);
            pool_.Free(p2);
        }

        // merge tail into head
        {
            intptr_t head = pool_.Malloc(kRoundSize * 2);
            intptr_t tail = pool_.Malloc(kRoundSize * 2);
            assert(p1 == head);
            pool_.Free(head);
            pool_.Free(tail);
            intptr_t p2 = pool_.Malloc(kRoundSize * 4);
            assert(p1 == p2);
            pool_.Free(p2);
        }
    }

    void TestFreeDifferentSize() {
        intptr_t p1 = pool_.Malloc(kRoundSize * 4);
        pool_.Free(p1);
        intptr_t p2 = pool_.Malloc(kRoundSize * 8);
        assert(p1 != p2);
    }

    // void TestFreeAllBlocks() {
    //     intptr_t p1 = pool_.Malloc(kRoundSize * 4);
    //     pool_.Free(p1);
    //     pool_.FreeAllBlocks();
    //     intptr_t p2 = pool_.Malloc(kRoundSize * 4);
    //     assert(p1 != p2);
    //     pool_.Free(p2);
    // }

    // def test_free_all_blocks_split(self):
    //     # do not free splitted blocks
    //     p = pool_.Malloc(kRoundSize * 4)
    //     del p
    //     head = pool_.Malloc(kRoundSize * 2)
    //     tail = pool_.Malloc(kRoundSize * 2)
    //     tailptr = tail.ptr
    //     del tail
    //     pool_.free_all_blocks()
    //     p = pool_.Malloc(kRoundSize * 2)
    //     assert(tailptr, p.ptr)
    //     del head

    // def test_free_all_blocks_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     ptr1 = p1.ptr
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 4)
    //         ptr2 = p2.ptr
    //         del p2
    //     pool_.free_all_blocks(stream=stream_module.Stream.null)
    //     p3 = pool_.Malloc(kRoundSize * 4)
    //     self.assertNotEqual(ptr1, p3.ptr)
    //     self.assertNotEqual(ptr2, p3.ptr)
    //     with self.stream:
    //         p4 = pool_.Malloc(kRoundSize * 4)
    //         self.assertNotEqual(ptr1, p4.ptr)
    //         assert(ptr2, p4.ptr)

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

    // def test_free_all_free(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     ptr1 = p1.ptr
    //     del p1
    //     with testing.assert_warns(DeprecationWarning):
    //         pool_.free_all_free()
    //     p2 = pool_.Malloc(kRoundSize * 4)
    //     self.assertNotEqual(ptr1, p2.ptr)

    // def test_used_bytes(self):
    //     p1 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 2, pool_.used_bytes())
    //     p2 = pool_.Malloc(kRoundSize * 4)
    //     assert(kRoundSize * 6, pool_.used_bytes())
    //     del p2
    //     assert(kRoundSize * 2, pool_.used_bytes())
    //     del p1
    //     assert(kRoundSize * 0, pool_.used_bytes())
    //     p3 = pool_.Malloc(kRoundSize * 1)
    //     assert(kRoundSize * 1, pool_.used_bytes())
    //     del p3

    // def test_used_bytes_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 2, pool_.used_bytes())
    //     del p2

    // def test_free_bytes(self):
    //     p1 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 0, pool_.free_bytes())
    //     p2 = pool_.Malloc(kRoundSize * 4)
    //     assert(kRoundSize * 0, pool_.free_bytes())
    //     del p2
    //     assert(kRoundSize * 4, pool_.free_bytes())
    //     del p1
    //     assert(kRoundSize * 6, pool_.free_bytes())
    //     p3 = pool_.Malloc(kRoundSize * 1)
    //     assert(kRoundSize * 5, pool_.free_bytes())
    //     del p3

    // def test_free_bytes_stream(self):
    //     p1 = pool_.Malloc(kRoundSize * 4)
    //     del p1
    //     with self.stream:
    //         p2 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 4, pool_.free_bytes())
    //     del p2

    // def test_total_bytes(self):
    //     p1 = pool_.Malloc(kRoundSize * 2)
    //     assert(kRoundSize * 2, pool_.total_bytes())
    //     p2 = pool_.Malloc(kRoundSize * 4)
    //     assert(kRoundSize * 6, pool_.total_bytes())
    //     del p1
    //     assert(kRoundSize * 6, pool_.total_bytes())
    //     del p2
    //     assert(kRoundSize * 6, pool_.total_bytes())
    //     p3 = pool_.Malloc(kRoundSize * 1)
    //     assert(kRoundSize * 6, pool_.total_bytes())
    //     del p3

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
    return 0;
}
