# frozen_string_literal: true

KERNEL_PARTS = %w[main reduce reduce_nan reduce_from math math_rare compare cast misc].freeze

def kernel_part_basename(type_name, part)
  part == "main" ? "#{type_name}_kernel" : "#{type_name}_kernel_#{part}"
end
