#ifndef RERDMFT_UTILS_DETHASH_H
#define RERDMFT_UTILS_DETHASH_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace rerdmft {

// Debugging aid for run-to-run reproducibility (env RERDMFT_DET_PROBE): prints a hash of the exact bits of an array to stderr as
// "[DET] label hash". Two processes that print different hashes for the same label first differ at that point of the pipeline.
inline bool detProbeEnabled() {
  static const bool on = std::getenv("RERDMFT_DET_PROBE") != nullptr;
  return on;
}

inline void detHash(const char* label, const void* data, std::size_t bytes) {
  if (!detProbeEnabled()) return;
  std::uint64_t h = 1469598103934665603ULL;
  const unsigned char* b = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * 1099511628211ULL;
  std::cerr << "[DET] " << label << " " << std::hex << (h & 0xffffffffffULL) << std::dec << "\n";
}

template <typename M>
inline void detHashMatrix(const char* label, const M& m) {
  if (!detProbeEnabled()) return;
  using V = std::remove_cv_t<std::remove_reference_t<decltype(*m.data())>>;
  detHash(label, m.data(), m.rows() * m.cols() * sizeof(V));
}

}  // namespace rerdmft

#endif
