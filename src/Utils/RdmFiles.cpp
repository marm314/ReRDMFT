#include "RdmFiles.h"

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace rerdmft {

namespace {

using cplx = std::complex<double>;

// Little-endian stream of fixed-size records (the layout of CC_RDM's RecordWriter).
class RecordFile {
 public:
  explicit RecordFile(const std::string& path) : f_(std::fopen(path.c_str(), "wb")) {
    if (f_ == nullptr) throw std::runtime_error("RDM files: cannot open " + path + " for writing");
    std::setvbuf(f_, nullptr, _IOFBF, std::size_t(1) << 22);
  }
  ~RecordFile() {
    if (f_ != nullptr) std::fclose(f_);
  }
  RecordFile(const RecordFile&) = delete;
  RecordFile& operator=(const RecordFile&) = delete;
  void put(const std::int32_t* idx, int n, double re, double im) {
    std::fwrite(idx, sizeof(std::int32_t), static_cast<std::size_t>(n), f_);
    std::fwrite(&re, sizeof(double), 1, f_);
    std::fwrite(&im, sizeof(double), 1, f_);
  }
  void close() {
    if (f_ != nullptr) std::fclose(f_);
    f_ = nullptr;
  }

 private:
  std::FILE* f_;
};

}  // namespace

RdmFilesResult writeMoRdmFiles(const RdmSource& src, const std::string& dm1_path, const std::string& dm2_path, double threshold,
                               std::ostream& log) {
  const std::size_t n = src.n;
  if (src.gamma1.rows() != n || src.gamma1.cols() != n || !src.slab) throw std::runtime_error("writeMoRdmFiles: inconsistent source");
  RdmFilesResult res;
  cplx e1{};
  {
    RecordFile w1(dm1_path);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) {
        const cplx v = src.gamma1(p, q);
        if (p == q) res.trace_dm1 += v.real();
        if (std::abs(v) < threshold) continue;
        const std::int32_t idx[2] = {static_cast<std::int32_t>(p + 1), static_cast<std::int32_t>(q + 1)};
        w1.put(idx, 2, v.real(), v.imag());
        ++res.n_dm1;
        if (src.h.rows() == n) e1 += src.h(p, q) * v;
      }
    const std::int32_t zero[2] = {0, 0};
    w1.put(zero, 2, 0.0, 0.0);
    w1.close();
  }
  res.e_onebody = e1.real();
  log << "  1-RDM written to " << dm1_path << " (" << res.n_dm1 << " elements), trace = " << std::fixed << std::setprecision(10)
      << res.trace_dm1 << "\n";

  RecordFile w2(dm2_path);
  std::vector<cplx> slab(n * n * n);
  for (std::size_t p = 0; p < n; ++p) {
    src.slab(p, slab.data());
    for (std::size_t q = 0; q < n; ++q) {
      if (src.antisymmetric && q <= p) continue;  // p<q only (antisymmetry gives the rest)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          if (src.antisymmetric && s <= r) continue;  // r<s only
          const cplx v = slab[(q * n + r) * n + s];
          // sum_pq Gamma_pqpq: the p<q, r<s half counts twice
          if (p == r && q == s) res.trace_dm2 += (src.antisymmetric ? 2.0 : 1.0) * v.real();
          if (std::abs(v) < threshold) continue;
          const std::int32_t idx[4] = {static_cast<std::int32_t>(p + 1), static_cast<std::int32_t>(q + 1),
                                       static_cast<std::int32_t>(r + 1), static_cast<std::int32_t>(s + 1)};
          w2.put(idx, 4, v.real(), v.imag());
          ++res.n_dm2;
        }
    }
  }
  const std::int32_t zero[4] = {0, 0, 0, 0};
  w2.put(zero, 4, 0.0, 0.0);
  w2.close();
  log << "  2-RDM written to " << dm2_path << " (" << res.n_dm2 << " elements" << (src.antisymmetric ? " with p<q, r<s" : ", all nonzero")
      << "), trace = " << std::fixed << std::setprecision(10) << res.trace_dm2 << "\n";
  return res;
}

}  // namespace rerdmft
