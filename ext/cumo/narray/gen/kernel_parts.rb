# frozen_string_literal: true

KERNEL_PARTS = %w[main reduce reduce_nan reduce_from math math_rare compare cast misc].freeze
MATH_TYPES = %w[hfloat bfloat sfloat dfloat scomplex dcomplex].freeze

def kernel_parts_of(type_name)
  return KERNEL_PARTS.first(4) if type_name == "robject"

  KERNEL_PARTS.reject do |part|
    (%w[math math_rare].include?(part) && !MATH_TYPES.include?(type_name)) ||
      (%w[compare misc].include?(part) && type_name == "bit")
  end
end

def kernel_part_basename(type_name, part)
  part == "main" ? "#{type_name}_kernel" : "#{type_name}_kernel_#{part}"
end
