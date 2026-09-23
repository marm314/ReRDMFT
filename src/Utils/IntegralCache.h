#ifndef RERDMFT_INTEGRALCACHE_H
#define RERDMFT_INTEGRALCACHE_H

#include <cstdint>
#include <string>
#include <vector>

#include "ElectronRepulsion.h"
#include "MolecularBasis.h"
#include "RkbTwoElectron.h"

namespace rerdmft {

// Bump whenever the on-disk cache file layout, OR the algorithm that
// produces a cached tensor (RkbTwoElectron.cpp, ElectronRepulsion.cpp),
// changes -- a cache built by an older version of this code must never
// be silently reused as if it were still correct. loadX below already
// refuses to load a file whose stored version does not match this one.
constexpr std::uint32_t kIntegralCacheFormatVersion = 1;

// Order-sensitive 64-bit hash over the concrete, already-built basis
// functions (element, center, angular momentum, exponents, contraction
// coefficients) -- NOT over the geometry/basis-file text, so it stays
// correct under any change to how the basis is parsed, ordered, or
// normalized without needing a separately-maintained invalidation rule.
// Two bases that hash equal are, short of an astronomically unlikely
// collision, the same basis.
//
// Notably, this (and hence every cache key below) does NOT depend on
// the speed of light: rkbCoefficients, and therefore every two-electron
// integral built from it, is purely a function of the Large/Small AO
// bases (RkbTransformation.h) -- so, e.g., water-c1000.inp
// and water-c100000.inp (same geometry and basis,
// different SPEED_OF_LIGHT) share one cache entry.
std::uint64_t basisFingerprint(const std::vector<BasisFunction>& basis);

// Combines two independent fingerprints (e.g. Large and Small basis)
// into one, order-sensitive (combine(a,b) != combine(b,a) in general).
std::uint64_t combineFingerprints(std::uint64_t a, std::uint64_t b);

// Fixed-width lowercase hex string, suitable for use in a filename.
std::string fingerprintToHex(std::uint64_t fingerprint);

// Loads a tensor previously written by the matching saveX function below
// from `path` into `out` and returns true, but ONLY if the file exists
// and its header's format version, fingerprint, and dimension all match
// exactly; otherwise returns false (a cold cache, a different
// molecule/basis, or a cache from a different code version) and leaves
// `out` untouched -- the caller should then build the tensor normally
// (and typically save it via saveX for next time).
bool loadPackedTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                                  std::size_t expected_dim, PackedTwoElectronTensor& out);
bool loadRkbTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                               std::size_t expected_dim, RkbTwoElectronTensor& out);

// Writes `tensor` to `path` (creating parent directories as needed) with
// a header recording the current format version, `fingerprint`, and
// dimension, so a later loadX call against the same path can recognize
// whether it is still valid.
//
// The file layout is a fixed-size POD header (magic/version/fingerprint/
// dim/count) followed by the tensor's raw stored values, written and
// read back with the host's native endianness and struct layout -- this
// is a same-machine, same-build avoid-recompute cache, not a portable
// interchange format.
void savePackedTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                                  const PackedTwoElectronTensor& tensor);
void saveRkbTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                               const RkbTwoElectronTensor& tensor);

}  // namespace rerdmft

#endif  // RERDMFT_INTEGRALCACHE_H
