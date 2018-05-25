module MkmfCu
  COLOR_CODES = {
    red: 31,
    green: 32,
    yellow: 33,
    blue: 34,
    magenta: 35,
    cyan: 36
  }

  def self.colorize(code, str)
    raise "#{color_code} is not supported" unless COLOR_CODES[code]
    "\e[#{COLOR_CODES[code]}m#{str}\e[0m"
  end
end
