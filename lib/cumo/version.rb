# frozen_string_literal: true

module Cumo
  # The version of the cumo gem.
  VERSION = "0.16.1"

  spec = Gem.loaded_specs["cumo"] if defined?(Gem)
  # The git revision the activated cumo gem was built from, such as
  # "v1.2.3", or nil when the loaded files are not that gem's, as with
  # ruby -Ilib.
  REVISION = (spec.metadata["cumo_revision"]&.dup&.freeze if spec && spec.full_gem_path == File.expand_path("../..", __dir__))
end
