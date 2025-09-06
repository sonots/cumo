require_relative "test_helper"

class BitTest < Test::Unit::TestCase
  dtype = Cumo::Bit

  test dtype do
    assert { dtype < Cumo::NArray }
  end

  procs = [
    [proc{|tp,a| tp[*a] },""],
    [proc{|tp,a| tp[*a][true] },"[true]"],
    [proc{|tp,a| tp[*a][0..-1] },"[0..-1]"]
  ]
  procs.each do |init, ref|

    test "#{dtype},[0,1,1,0,1,0,0,1]#{ref}" do
      src = [0,1,1,0,1,0,0,1]
      n = src.size
      a = init.call(dtype, src)

      assert { a == src }
      assert { (a & 0) == [0] * n }
      assert { (a & 1) == src }
      assert { (a | 0) == src }
      assert { (a | 1) == [1] * n }
      assert { (a ^ 0) == src.map {|x| x ^ 0 } }
      assert { (a ^ 1) == src.map {|x| x ^ 1 } }
      assert { ~a == src.map {|x| 1 - x } }

      assert { a.count_true == 4 }
      assert { a.count_false == 4 }
      assert { a.where == [1,2,4,7] }
      assert { a.where2 == [[1,2,4,7], [0,3,5,6]] }
      # TODO(sonots): FIX ME
      # assert { a.mask(Cumo::DFloat[1,2,3,4,5,6,7,8]) == [2,3,5,8] }
      assert { !a.all? }
      assert { a.any? }
      assert { !a.none? }
    end
  end

  procs = [
    [proc{|tp,a| tp[*a] },""],
    [proc{|tp,a| tp[*a][true,0..-1] },"[true,true]"],
  ]
  procs.each do |init, ref|

    test "#{dtype},[[0,1,1,0],[1,0,0,1]]#{ref}" do
      src = [[0,1,1,0],[1,0,0,1]]
      a = init.call(dtype, src)

      assert { a[5] == 0 }
      assert { a[-1] == 1 }
      assert { a[1,0] == src[1][0] }
      assert { a[1,1] == src[1][1] }
      assert { a[1,2] == src[1][2] }
      assert { a[3..4] == [0,1] }
      assert { a[0,1..2] == [1,1] }
      assert { a[0,:*] == src[0] }
      assert { a[1,:*] == src[1] }
      assert { a[:*,1] == [src[0][1], src[1][1]] }

      assert { a.count_true == 4 }
      assert { a.count_false == 4 }
      assert { a.where == [1,2,4,7] }
      assert { a.where2 == [[1,2,4,7],[0,3,5,6]] }
      # TODO(sonots): FIX ME
      # assert { a.mask(Cumo::DFloat[[1,2,3,4],[5,6,7,8]]) == [2,3,5,8] }
      assert { !a.all? }
      assert { a.any? }
      assert { !a.none? }
    end

    test "#{dtype},[[0,1,1,0],[1,0,0,1]]#{ref},aset[]=" do
      src = [[0,1,1,0],[1,0,0,1]]

      a = init.call(dtype, src)
      a[5] = 1
      assert { a[5] == 1 }

      a = init.call(dtype, src)
      a[-1] = 0
      assert { a[-1] == 0 }

      a = init.call(dtype, src)
      a[1,0] = 0
      assert { a[1,0] == 0 }

      a = init.call(dtype, src)
      a[1,1] = 1
      assert { a[1,1] == 1}

      a = init.call(dtype, src)
      a[1,2] = 1
      assert { a[1,2] == 1 }

      a = init.call(dtype, src)
      a[3..4] = [1,0]
      assert { a[3..4] == [1,0] }

      a = init.call(dtype, src)
      a[0,1..2] = [0,0]
      assert { a[0,1..2] == [0,0] }

      a = init.call(dtype, src)
      a[0,:*] = [1,0,0,1]
      assert { a[0,:*] == [1,0,0,1] }

      a = init.call(dtype, src)
      a[1,:*] = [0,1,1,0]
      assert { a[1,:*] == [0,1,1,0] }

      a = init.call(dtype, src)
      a[:*,1] = [0,1]
      assert { a[:*,1] == [0,1] }

      a = init.call(dtype, src)
      a[5] = dtype.cast(1)
      assert { a[5] == 1 }
      assert { a[5] == dtype.cast(1) }

      a = init.call(dtype, src)
      a[1,0] = dtype.cast(0)
      assert { a[1,0] == 0 }
      assert { a[1,0] == dtype.cast(0) }

      a = init.call(dtype, src)
      a[3..4] = dtype.cast([1,0])
      assert { a[3..4] == [1,0] }
      assert { a[3..4] == dtype.cast([1,0]) }

      a = init.call(dtype, src)
      a[:*,1] = dtype.cast([0,1])
      assert { a[:*,1] == [0,1] }
      assert { a[:*,1] == dtype.cast([0,1]) }
    end
  end
end
