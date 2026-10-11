# frozen_string_literal: true

module Cumo
  revision_file = File.join(__dir__, "REVISION")
  REVISION =
    if File.exist?(File.expand_path("../../.git", __dir__))
      IO.popen(["git", "-C", File.expand_path("../..", __dir__), "describe", "--tags", "--match", "v[0-9]*", "--dirty", "--always"],
               err: File::NULL, &:read).strip
    elsif File.exist?(revision_file)
      File.read(revision_file).strip
    end
end
