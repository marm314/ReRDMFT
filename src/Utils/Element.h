#ifndef RERDMFT_ELEMENT_H
#define RERDMFT_ELEMENT_H

#include <string>

namespace rerdmft {

// Atomic number (nuclear charge Z) for an element symbol (case-insensitive),
// covering the full periodic table (H=1 .. Og=118). Throws std::runtime_error
// if the symbol is not recognized.
int atomicNumber(const std::string& symbol);

}  // namespace rerdmft

#endif  // RERDMFT_ELEMENT_H
