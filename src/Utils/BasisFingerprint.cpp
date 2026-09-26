#include "BasisFingerprint.h"

namespace rerdmft {

namespace {

// 64-bit FNV-1a: simple and dependency-free.
constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

std::uint64_t fnv1aUpdateBytes(std::uint64_t hash, const void* data, std::size_t n) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < n; ++i) {
    hash ^= bytes[i];
    hash *= kFnvPrime;
  }
  return hash;
}

template <typename T>
std::uint64_t fnv1aUpdateValue(std::uint64_t hash, const T& value) {
  return fnv1aUpdateBytes(hash, &value, sizeof(T));
}

}  // namespace

std::uint64_t basisFingerprint(const std::vector<BasisFunction>& basis) {
  std::uint64_t hash = kFnvOffsetBasis;
  for (const auto& fn : basis) {
    hash = fnv1aUpdateBytes(hash, fn.element.data(), fn.element.size());
    hash = fnv1aUpdateValue(hash, fn.x);
    hash = fnv1aUpdateValue(hash, fn.y);
    hash = fnv1aUpdateValue(hash, fn.z);
    hash = fnv1aUpdateValue(hash, fn.l);
    hash = fnv1aUpdateValue(hash, fn.cartesian.lx);
    hash = fnv1aUpdateValue(hash, fn.cartesian.ly);
    hash = fnv1aUpdateValue(hash, fn.cartesian.lz);
    hash = fnv1aUpdateValue(hash, fn.exponents.size());
    for (double exponent : fn.exponents) hash = fnv1aUpdateValue(hash, exponent);
    for (double coefficient : fn.coefficients) hash = fnv1aUpdateValue(hash, coefficient);
  }
  return hash;
}

}  // namespace rerdmft
