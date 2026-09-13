#ifndef RERDMFT_BASISSET_H
#define RERDMFT_BASISSET_H

#include <map>
#include <string>
#include <vector>

#include "Shell.h"

namespace rerdmft {

// Reads a Gaussian-format basis set file and stores, for each element
// symbol, the list of contracted shells that define its basis. This is a
// per-element template: it is not yet tied to any atomic position.
class BasisSet {
 public:
  void read(const std::string& filename);

  bool hasElement(const std::string& symbol) const;
  const std::vector<Shell>& shellsForElement(const std::string& symbol) const;

 private:
  std::map<std::string, std::vector<Shell>> shells_by_element_;
};

}  // namespace rerdmft

#endif  // RERDMFT_BASISSET_H
