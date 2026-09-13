#ifndef RERDMFT_STRINGUTILS_H
#define RERDMFT_STRINGUTILS_H

#include <string>

namespace rerdmft {

std::string toUpper(std::string s);

// Strips a trailing '#' or '!' comment and surrounding whitespace from a line.
std::string stripComment(const std::string& line);

// Parses a double that may use Fortran-style 'D' exponent notation
// (e.g. "1.234D+00") in addition to the standard 'E'/'e' notation.
double parseFortranDouble(const std::string& token);

}  // namespace rerdmft

#endif  // RERDMFT_STRINGUTILS_H
