#ifndef RERDMFT_CC_CCTENSOR_H
#define RERDMFT_CC_CCTENSOR_H

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace rerdmft {
namespace cc {

using cplx = std::complex<double>;

// Dense row-major complex tensor of any rank, with permutations and an einsum evaluated as one ZGEMM per contraction.
struct Tensor {
  std::vector<std::size_t> dims;
  std::vector<cplx> d;
  Tensor() = default;
  explicit Tensor(std::vector<std::size_t> dm) : dims(std::move(dm)) {
    std::size_t n = 1;
    for (auto x : dims) n *= x;
    d.assign(n, cplx{});
  }
  std::size_t size() const { return d.size(); }
};

// out axis k = in axis perm[k].
Tensor permute(const Tensor& a, const std::vector<int>& perm);
Tensor conjugated(const Tensor& a);
void axpy(Tensor& y, const Tensor& x, cplx s);  // y += s x
Tensor operator+(Tensor a, const Tensor& b);
Tensor operator-(Tensor a, const Tensor& b);
cplx dotNoConj(const Tensor& a, const Tensor& b);
double maxAbs(const Tensor& t);

// einsum over two operands: every label shared by A and B is summed (and must not appear in the output); every other
// label must appear in the output. alpha scales the result.
Tensor einsum(const std::string& spec, const Tensor& A, const Tensor& B, cplx alpha = 1.0);
// Any number (>= 1) of operands, contracted pairwise from left to right.
Tensor einsumN(const std::string& spec, const std::vector<const Tensor*>& ops, cplx alpha = 1.0);

// X - X with the first two axes swapped / the last two axes swapped (rank-4 tensors).
Tensor antisymIJ(const Tensor& x);
Tensor antisymAB(const Tensor& x);

// The sub-tensor with index `index` fixed along `axis` (rank decreases by one).
Tensor takeAxis(const Tensor& a, std::size_t axis, std::size_t index);
// Adds `src` into the sub-tensor of `dst` with index `index` fixed along `axis`.
void addIntoAxis(Tensor& dst, std::size_t axis, std::size_t index, const Tensor& src);

}  // namespace cc
}  // namespace rerdmft

#endif  // RERDMFT_CC_CCTENSOR_H
