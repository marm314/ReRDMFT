#include "IntegralCache.h"

#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace rerdmft {

namespace {

// 64-bit FNV-1a: simple, dependency-free, and more than adequate for a
// same-machine cache-key hash (collisions would need to be astronomically
// unlucky to matter here).
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

// Fixed-layout header written verbatim at the start of every cache file;
// see IntegralCache.h's saveX doc comment for why this is a same-machine
// format, not a portable one.
struct CacheHeader {
  char magic[4];
  std::uint32_t version;
  std::uint64_t fingerprint;
  std::uint64_t dim;
  std::uint64_t stored_count;
};

// Reads `path` into `data_out`/`dim_out`, returning true only if it is a
// well-formed cache file whose header matches `fingerprint`/`expected_dim`
// under the CURRENT format version -- see loadPackedTwoElectronTensor's
// and loadRkbTwoElectronTensor's shared contract in IntegralCache.h.
template <typename T>
bool loadTensorFile(const std::string& path, std::uint64_t fingerprint, std::size_t expected_dim,
                     std::vector<T>& data_out, std::size_t& dim_out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  CacheHeader header{};
  in.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!in || in.gcount() != static_cast<std::streamsize>(sizeof(header))) return false;
  if (std::memcmp(header.magic, "RERI", 4) != 0) return false;
  if (header.version != kIntegralCacheFormatVersion) return false;
  if (header.fingerprint != fingerprint) return false;
  if (header.dim != static_cast<std::uint64_t>(expected_dim)) return false;

  std::vector<T> data(header.stored_count);
  const auto payload_bytes = static_cast<std::streamsize>(header.stored_count * sizeof(T));
  in.read(reinterpret_cast<char*>(data.data()), payload_bytes);
  if (!in || in.gcount() != payload_bytes) return false;

  data_out = std::move(data);
  dim_out = static_cast<std::size_t>(header.dim);
  return true;
}

// Writes `data` to `path` (creating parent directories as needed) behind
// a CacheHeader tagged with the current format version.
template <typename T>
void saveTensorFile(const std::string& path, std::uint64_t fingerprint, std::size_t dim,
                     const std::vector<T>& data) {
  const std::filesystem::path fs_path(path);
  if (fs_path.has_parent_path()) {
    std::filesystem::create_directories(fs_path.parent_path());
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("IntegralCache: could not open '" + path + "' for writing");
  }

  CacheHeader header{};
  std::memcpy(header.magic, "RERI", 4);
  header.version = kIntegralCacheFormatVersion;
  header.fingerprint = fingerprint;
  header.dim = static_cast<std::uint64_t>(dim);
  header.stored_count = static_cast<std::uint64_t>(data.size());

  out.write(reinterpret_cast<const char*>(&header), sizeof(header));
  out.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));
  if (!out) {
    throw std::runtime_error("IntegralCache: failed writing '" + path + "'");
  }
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

std::uint64_t combineFingerprints(std::uint64_t a, std::uint64_t b) {
  return fnv1aUpdateValue(fnv1aUpdateValue(kFnvOffsetBasis, a), b);
}

std::string fingerprintToHex(std::uint64_t fingerprint) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << fingerprint;
  return oss.str();
}

bool loadPackedTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                                  std::size_t expected_dim, PackedTwoElectronTensor& out) {
  std::vector<double> data;
  std::size_t dim = 0;
  if (!loadTensorFile(path, fingerprint, expected_dim, data, dim)) return false;
  out = PackedTwoElectronTensor(dim, std::move(data));
  return true;
}

void savePackedTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                                  const PackedTwoElectronTensor& tensor) {
  saveTensorFile(path, fingerprint, tensor.dim(), tensor.rawData());
}

bool loadRkbTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                               std::size_t expected_dim, RkbTwoElectronTensor& out) {
  std::vector<std::complex<double>> data;
  std::size_t dim = 0;
  if (!loadTensorFile(path, fingerprint, expected_dim, data, dim)) return false;
  out = RkbTwoElectronTensor(dim, std::move(data));
  return true;
}

void saveRkbTwoElectronTensor(const std::string& path, std::uint64_t fingerprint,
                               const RkbTwoElectronTensor& tensor) {
  saveTensorFile(path, fingerprint, tensor.dim(), tensor.rawData());
}

}  // namespace rerdmft
