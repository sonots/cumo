# frozen_string_literal: true

# The driver loads a whole module into host memory the first time any kernel in
# it runs, so a type's kernels are split into parts that are each built into a
# module of their own: a program that never reduces, or never asks for the
# nan-aware or mixed-type forms, never pays for them.
KERNEL_PARTS = %w[main reduce reduce_nan reduce_from].freeze

def kernel_part_basename(type_name, part)
  part == "main" ? "#{type_name}_kernel" : "#{type_name}_#{part}_kernel"
end
