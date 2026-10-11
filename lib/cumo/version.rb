# frozen_string_literal: true

module Cumo
  VERSION = "0.13.0"

  spec = Gem.loaded_specs["cumo"] if defined?(Gem)
  REVISION = (spec.metadata["cumo_revision"]&.dup&.freeze if spec && spec.full_gem_path == File.expand_path("../..", __dir__))
end
