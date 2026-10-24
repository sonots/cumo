# frozen_string_literal: true

KERNEL_PARTS = %w[main reduce reduce_nan reduce_from reduce_rare math math_rare compare cast misc sort].freeze
MATH_TYPES = %w[hfloat bfloat sfloat dfloat scomplex dcomplex].freeze
SORT_TYPES = %w[int8 int16 int32 int64 uint8 uint16 uint32 uint64 hfloat bfloat sfloat dfloat].freeze

def kernel_parts_of(type_name)
  return %w[main reduce reduce_nan reduce_from] if type_name == "robject"

  KERNEL_PARTS.reject do |part|
    (%w[math math_rare].include?(part) && !MATH_TYPES.include?(type_name)) ||
      (%w[compare misc reduce_rare].include?(part) && type_name == "bit") ||
      (part == "sort" && !SORT_TYPES.include?(type_name))
  end
end

def kernel_part_basename(type_name, part)
  part == "main" ? "#{type_name}_kernel" : "#{type_name}_kernel_#{part}"
end
