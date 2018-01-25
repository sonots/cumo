require_relative "test_helper"

class NArrayTest < Test::Unit::TestCase
  types = [
    Cumo::DFloat,
    Cumo::SFloat,
    Cumo::DComplex,
    Cumo::SComplex,
    Cumo::Int64,
    Cumo::Int32,
    Cumo::Int16,
    Cumo::Int8,
    Cumo::UInt64,
    Cumo::UInt32,
    Cumo::UInt16,
    Cumo::UInt8,
  ]
  float_types = [
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  if ENV['ONLY']
    types.select!{|type| type.to_s.downcase.include?(ENV['ONLY'].downcase) }
    float_types.select!{|type| type.to_s.downcase.include?(ENV['ONLY'].downcase) }
  end

  types.each do |dtype|
    test dtype do
      assert { dtype < Cumo::NArray }
    end

    procs = [
      [proc{|tp,a| tp[*a] },""],
      [proc{|tp,a| tp[*a][true] },"[true]"],
      [proc{|tp,a| tp[*a][0..-1] },"[0..-1]"]
    ]
    procs.each do |init, ref|

      test "#{dtype},[1,2,3,5,7,11]#{ref}" do
        src = [1,2,3,5,7,11]
        a = init.call(dtype, src)

        assert { a.is_a?(dtype) }
        assert { a.size == 6 }
        assert { a.ndim == 1 }
        assert { a.shape == [6] }
        assert { !a.inplace? }
        assert { a.row_major? }
        assert { !a.column_major? }
        assert { a.host_order? }
        assert { !a.byte_swapped? }
        assert { a == [1,2,3,5,7,11] }
        assert { a.to_a == [1,2,3,5,7,11] }
        assert { a.to_a.is_a?(Array) }
        assert { a.dup == a }
        assert { a.clone == a }
        assert { a.dup.object_id != a.object_id }
        assert { a.clone.object_id != a.object_id }

        assert { a.eq([1,1,3,3,7,7]) == [1,0,1,0,1,0] }
        assert { a[3..4] == [5,7] }
        assert { a[5] == 11 }
        assert { a[-1] == 11 }
        assert { a[[4,3,0,1,5,2]] == [7,5,1,2,11,3] }
        assert { a.sum == 29 }
        if float_types.include?(dtype)
          assert { a.mean == 29.0/6 }
          assert { a.var == 13.766666666666669 }
          assert { a.stddev == 3.710345895825168 }
          assert { a.rms == 5.901977069875258 }
        end
        assert { a.dup.fill(12) == [12]*6 }
        assert { (a + 1) == [2,3,4,6,8,12] }
        assert { (a - 1) == [0,1,2,4,6,10] }
        assert { (a * 3) == [3,6,9,15,21,33] }
        assert { (a / 0.5) == [2,4,6,10,14,22] }
        assert { (-a) == [-1,-2,-3,-5,-7,-11] }
        assert { (a ** 2) == [1,4,9,25,49,121] }
        assert { a.swap_byte.swap_byte == [1,2,3,5,7,11] }
        if dtype == Cumo::DComplex || dtype == Cumo::SComplex
          assert { a.real == src }
          assert { a.imag == [0]*6 }
          assert { a.conj == src }
          assert { a.angle == [0]*6 }
        else
          assert { a.min == 1 }
          assert { a.max == 11 }
          assert { a.min_index == 0 }
          assert { a.max_index == 5 }
          assert { (a >= 3) == [0,0,1,1,1,1] }
          assert { (a >  3) == [0,0,0,1,1,1] }
          assert { (a <= 3) == [1,1,1,0,0,0] }
          assert { (a <  3) == [1,1,0,0,0,0] }
          assert { (a.eq 3) == [0,0,1,0,0,0] }
          assert { a.sort == src }
          assert { a.sort_index == (0..5).to_a }
          assert { a.median == 4 }
        end
      end
    end

    test "#{dtype},[1..4]" do
      assert { dtype[1..4] == [1,2,3,4] }
    end

    procs2 = [
      [proc{|tp,src| tp[*src] },""],
      [proc{|tp,src| tp[*src][true,true] },"[true,true]"],
      [proc{|tp,src| tp[*src][0..-1,0..-1] },"[0..-1,0..-1]"]
    ]

    procs2.each do |init, ref|

      test "#{dtype},[[1,2,3],[5,7,11]]#{ref}" do
        src = [[1,2,3],[5,7,11]]
        a = init.call(dtype, src)

        assert { a.is_a?(dtype) }
        assert { a.size == 6 }
        assert { a.ndim == 2 }
        assert { a.shape == [2,3] }
        assert { !a.inplace? }
        assert { a.row_major? }
        assert { !a.column_major? }
        assert { a.host_order? }
        assert { !a.byte_swapped? }
        assert { a == src }
        assert { a.to_a == src }
        assert { a.to_a.is_a?(Array) }

        assert { a.eq([[1,1,3],[3,7,7]]) == [[1,0,1],[0,1,0]] }
        assert { a[5] == 11 }
        assert { a[-1] == 11 }
        assert { a[1,0] == src[1][0] }
        assert { a[1,1] == src[1][1] }
        assert { a[1,2] == src[1][2] }
        assert { a[3..4] == [5,7] }
        assert { a[0,1..2] == [2,3] }
        assert { a[0,:*] == src[0] }
        assert { a[1,:*] == src[1] }
        assert { a[:*,1] == [src[0][1],src[1][1]] }
        assert { a[true,[2,0,1]] == [[3,1,2],[11,5,7]] }
        assert { a.reshape(3,2) == [[1,2],[3,5],[7,11]] }
        assert { a.reshape(3,nil) == [[1,2],[3,5],[7,11]] }
        assert { a.reshape(nil,2) == [[1,2],[3,5],[7,11]] }
        assert { a.transpose == [[1,5],[2,7],[3,11]] }
        assert { a.transpose(1,0) == [[1,5],[2,7],[3,11]] }

        assert { a.sum == 29 }
        assert { a.sum(0) == [6, 9, 14] }
        assert { a.sum(1) == [6, 23] }
        if float_types.include?(dtype)
          assert { a.mean == 29.0/6 }
          assert { a.mean(0) == [3, 4.5, 7] }
          assert { a.mean(1) == [2, 23.0/3] }
        end
        if dtype == Cumo::DComplex || dtype == Cumo::SComplex
          assert { a.real == src }
          assert { a.imag == [[0]*3]*2 }
          assert { a.conj == src }
          assert { a.angle == [[0]*3]*2 }
        else
          assert { a.min == 1 }
          assert { a.max == 11 }
          assert { (a >= 3) == [[0,0,1],[1,1,1]] }
          assert { (a >  3) == [[0,0,0],[1,1,1]] }
          assert { (a <= 3) == [[1,1,1],[0,0,0]] }
          assert { (a <  3) == [[1,1,0],[0,0,0]] }
          assert { (a.eq 3) == [[0,0,1],[0,0,0]] }
          assert { a.sort == src }
          assert { a.sort_index == [[0,1,2],[3,4,5]] }
        end
        assert { a.dup.fill(12) == [[12]*3]*2 }
        assert { (a + 1) == [[2,3,4],[6,8,12]] }
        assert { (a + [1,2,3]) == [[2,4,6],[6,9,14]] }
        assert { (a - 1) == [[0,1,2],[4,6,10]] }
        assert { (a - [1,2,3]) == [[0,0,0],[4,5,8]] }
        assert { (a * 3) == [[3,6,9],[15,21,33]] }
        assert { (a * [1,2,3]) == [[1,4,9],[5,14,33]] }
        assert { (a / 0.5) == [[2,4,6],[10,14,22]] }
        assert { (-a) == [[-1,-2,-3],[-5,-7,-11]] }
        assert { (a ** 2) == [[1,4,9],[25,49,121]] }
        assert { (dtype[[1,0],[0,1]].dot dtype[[4,1],[2,2]]) == [[4,1],[2,2]] }
        assert { a.swap_byte.swap_byte == src }
      end

    end

    test "#{dtype},[[[1,2],[3,4]],[[5,6],[7,8]]]" do
      a = dtype[[[1,2],[3,4]],[[5,6],[7,8]]]

      assert { a[0, 1, 1] == 4 }
      assert { a[:rest] == a }
      assert { a[0, :rest] == [[1,2],[3,4]] }
      assert { a[0, false] == [[1,2],[3,4]] }
      assert { a[0, 1, :rest] == [3,4] }
      assert { a[0, 1, false] == [3,4] }
      assert { a[:rest, 0] == [[1,3],[5,7]] }
      assert { a[:rest, 0, 1] == [2,6] }
      assert { a[1, :rest, 0] == [5,7] }
      assert { a[1, 1, :rest, 0] == 7 }
      assert_raise(IndexError) { a[1, 1, 1, 1, :rest] }
      assert_raise(IndexError) { a[1, 1, 1, :rest, 1] }
      assert_raise(IndexError) { a[:rest, 1, :rest, 0] }
    end

    sub_test_case "#{dtype}, #mulsum" do
      test "vector.mulsum(vector)" do
        a = dtype[1..3]
        b = dtype[2..4]
        assert { a.mulsum(b) == (1*2 + 2*3 + 3*4) }
      end

      if [Cumo::DComplex, Cumo::SComplex, Cumo::DFloat, Cumo::SFloat].include?(dtype)
        test "vector.mulsum(vector, nan: true)" do
          a = dtype[1..3]
          a[0] = 0.0/0/0
          b = dtype[2..4]
          assert { a.mulsum(b, nan: true) == (0 + 2*3 + 3*4) }
        end
      end
    end

    sub_test_case "#{dtype}, #dot" do
      test "vector.dot(vector)" do
        a = dtype[1..3]
        b = dtype[2..4]
        assert { a.dot(b) == (1*2 + 2*3 + 3*4) }
      end
      test "matrix.dot(vector)" do
        a = dtype[1..6].reshape(3,2)
        b = dtype[1..2]
        assert { a.dot(b) == [5, 11, 17] }
      end
      test "vector.dot(matrix)" do
        a = dtype[1..2]
        b = dtype[1..6].reshape(2,3)
        assert { a.dot(b) == [9, 12, 15] }
      end
      test "matrix.dot(matrix)" do
        a = dtype[1..6].reshape(3,2)
        b = dtype[1..6].reshape(2,3)
        assert { a.dot(b) == [[9, 12, 15], [19, 26, 33], [29, 40, 51]] }
        assert { b.dot(a) == [[22, 28], [49, 64]] }
      end
      test "matrix.dot(matrix) with incorrect shape" do
        a = dtype[1..6].reshape(3,2)
        b = dtype[1..9].reshape(3,3)
        assert_raise(Cumo::NArray::ShapeError) { a.dot(b) }
      end
    end

    test "#{dtype},eye" do
      assert { dtype.new(3, 3).eye(1) == [[1,0,0],[0,1,0],[0,0,1]] }
      assert { dtype.new(3, 3).eye(2) == [[2,0,0],[0,2,0],[0,0,2]] }
      assert { dtype.new(3, 3).eye(1, 1) == [[0,1,0],[0,0,1],[0,0,0]] }
      assert { dtype.new(3, 3).eye(1, -1) == [[0,0,0],[1,0,0],[0,1,0]] }
      assert { dtype.new(2, 2, 2).eye(1) == [[[1,0],[0,1]],[[1,0],[0,1]]] }
      assert { dtype.new(3, 1).eye(1) == [[1],[0],[0]] }
      assert { dtype.new(1, 3).eye(1) == [[1,0,0]] }
      assert { dtype.eye(3) == [[1,0,0],[0,1,0],[0,0,1]] }
      assert { dtype.eye(3, 1) == [[1],[0],[0]] }
      assert { dtype.eye(1, 3) == [[1,0,0]] }
    end
  end
end
