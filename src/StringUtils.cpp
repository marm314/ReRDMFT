#include "StringUtils.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace rerdmft {

std::string toUpper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                  [](unsigned char c) { return std::toupper(c); });
  return s;
}

std::string stripComment(const std::string& line) {
  const auto comment_pos = line.find_first_of("#!");
  std::string result = line.substr(0, comment_pos);
  const auto begin = result.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const auto end = result.find_last_not_of(" \t\r\n");
  return result.substr(begin, end - begin + 1);
}

double parseFortranDouble(const std::string& token) {
  std::string fixed = token;
  for (char& c : fixed) {
    if (c == 'D' || c == 'd') c = 'e';
  }
  try {
    return std::stod(fixed);
  } catch (const std::exception&) {
    throw std::runtime_error("could not parse floating point value '" + token +
                              "'");
  }
}

}  // namespace rerdmft
