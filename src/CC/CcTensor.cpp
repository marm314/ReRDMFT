#include "CcTensor.h"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <stdexcept>

namespace rerdmft {
namespace cc {

// OpenMP regions are entered only above this many elements: OpenMP and OpenBLAS each keep a spinning thread pool, and
// alternating between them for small loops oversubscribes the machine (a 52-spinor CCSD ran 7x slower, a 22-spinor
// one 100x); the permutation loops are memory-bound, so the large ones are the only ones worth a region.
constexpr std::size_t kOmpMinWork = std::size_t(1) << 22;

// ---------------------------------------------------------------------------------------------------------
// Dense tensors and a two-operand einsum on top of ZGEMM
// ---------------------------------------------------------------------------------------------------------

Tensor permute(const Tensor& a, const std::vector<int>& perm) {
  const std::size_t r = perm.size();
  std::vector<std::size_t> odims(r), istr(r, 1);
  for (std::size_t k = r; k-- > 1;) istr[k - 1] = istr[k] * a.dims[k];
  std::vector<std::size_t> ostr_in(r);  // input stride of each output axis
  for (std::size_t k = 0; k < r; ++k) {
    odims[k] = a.dims[static_cast<std::size_t>(perm[k])];
    ostr_in[k] = istr[static_cast<std::size_t>(perm[k])];
  }
  Tensor out(odims);
  if (out.size() == 0) return out;
  const std::size_t last = odims[r - 1], last_stride = ostr_in[r - 1];
  const std::size_t outer = out.size() / last;
#pragma omp parallel for schedule(static) if (out.size() > kOmpMinWork)
  for (std::ptrdiff_t o = 0; o < static_cast<std::ptrdiff_t>(outer); ++o) {
    std::size_t rem = static_cast<std::size_t>(o), off = 0;
    for (std::size_t k = r - 1; k-- > 0;) {
      off += (rem % odims[k]) * ostr_in[k];
      rem /= odims[k];
    }
    const cplx* src = a.d.data() + off;
    cplx* dst = out.d.data() + static_cast<std::size_t>(o) * last;
    for (std::size_t x = 0; x < last; ++x) dst[x] = src[x * last_stride];
  }
  return out;
}

Tensor conjugated(const Tensor& a) {
  Tensor out = a;
#pragma omp parallel for schedule(static) if (out.size() > kOmpMinWork)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(out.size()); ++i) out.d[static_cast<std::size_t>(i)] = std::conj(out.d[static_cast<std::size_t>(i)]);
  return out;
}

// y += s * x
void axpy(Tensor& y, const Tensor& x, cplx s) {
  if (y.size() != x.size()) throw std::logic_error("UCCSD axpy: size mismatch");
#pragma omp parallel for schedule(static) if (y.size() > kOmpMinWork)
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(y.size()); ++i) y.d[static_cast<std::size_t>(i)] += s * x.d[static_cast<std::size_t>(i)];
}

Tensor operator+(Tensor a, const Tensor& b) {
  axpy(a, b, 1.0);
  return a;
}
Tensor operator-(Tensor a, const Tensor& b) {
  axpy(a, b, -1.0);
  return a;
}

cplx dotNoConj(const Tensor& a, const Tensor& b) {
  cplx s{};
  for (std::size_t i = 0; i < a.size(); ++i) s += a.d[i] * b.d[i];
  return s;
}

static bool isIdentity(const std::vector<int>& p) {
  for (std::size_t k = 0; k < p.size(); ++k)
    if (p[k] != static_cast<int>(k)) return false;
  return true;
}

Tensor einsum(const std::string& spec, const Tensor& A, const Tensor& B, cplx alpha) {
  const auto comma = spec.find(',');
  const auto arrow = spec.find("->");
  const std::string la = spec.substr(0, comma), lb = spec.substr(comma + 1, arrow - comma - 1), lo = spec.substr(arrow + 2);
  if (la.size() != A.dims.size() || lb.size() != B.dims.size()) throw std::logic_error("UCCSD einsum: rank mismatch in " + spec);
  std::size_t label_dim[128] = {0};
  const auto setDim = [&](const std::string& l, const Tensor& t) {
    for (std::size_t k = 0; k < l.size(); ++k) {
      auto& d = label_dim[static_cast<int>(l[k])];
      if (d != 0 && d != t.dims[k]) throw std::logic_error("UCCSD einsum: dimension mismatch in " + spec);
      d = t.dims[k];
    }
  };
  setDim(la, A);
  setDim(lb, B);
  std::string a_only, b_only, sh;
  for (char c : la) {
    const bool in_b = lb.find(c) != std::string::npos, in_o = lo.find(c) != std::string::npos;
    if (in_b && in_o) throw std::logic_error("UCCSD einsum: batch label in " + spec);
    if (in_b) sh += c;
    else if (in_o) a_only += c;
    else throw std::logic_error("UCCSD einsum: unsummed label in " + spec);
  }
  for (char c : lb) {
    const bool in_a = la.find(c) != std::string::npos, in_o = lo.find(c) != std::string::npos;
    if (!in_a) {
      if (!in_o) throw std::logic_error("UCCSD einsum: unsummed label in " + spec);
      b_only += c;
    }
  }
  const auto prodOf = [&](const std::string& l) {
    std::size_t p = 1;
    for (char c : l) p *= label_dim[static_cast<int>(c)];
    return p;
  };
  const std::size_t M = prodOf(a_only), N = prodOf(b_only), K = prodOf(sh);
  const std::string lc = a_only + b_only;
  std::vector<std::size_t> cdims;
  for (char c : lc) cdims.push_back(label_dim[static_cast<int>(c)]);
  Tensor C(cdims);
  if (M * N > 0 && K > 0) {
    const auto permFor = [](const std::string& have, const std::string& want) {
      std::vector<int> p;
      for (char c : want) p.push_back(static_cast<int>(have.find(c)));
      return p;
    };
    const cplx* a_ptr = A.d.data();
    const cplx* b_ptr = B.d.data();
    Tensor a_perm, b_perm;
    CBLAS_TRANSPOSE ta = CblasNoTrans, tb = CblasNoTrans;
    int lda = static_cast<int>(K), ldb = static_cast<int>(N);
    if (la == a_only + sh) {
    } else if (la == sh + a_only) {
      ta = CblasTrans;
      lda = static_cast<int>(M);
    } else {
      a_perm = permute(A, permFor(la, a_only + sh));
      a_ptr = a_perm.d.data();
    }
    if (lb == sh + b_only) {
    } else if (lb == b_only + sh) {
      tb = CblasTrans;
      ldb = static_cast<int>(K);
    } else {
      b_perm = permute(B, permFor(lb, sh + b_only));
      b_ptr = b_perm.d.data();
    }
    const cplx zero(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, ta, tb, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), &alpha, a_ptr, lda, b_ptr, ldb,
                &zero, C.d.data(), static_cast<int>(N));
  }
  if (lc == lo) return C;
  std::vector<int> p;
  for (char c : lo) p.push_back(static_cast<int>(lc.find(c)));
  return isIdentity(p) ? C : permute(C, p);
}

// X - X with the first two axes swapped / the last two axes swapped.
Tensor antisymIJ(const Tensor& x) { return x - permute(x, {1, 0, 2, 3}); }
Tensor antisymAB(const Tensor& x) { return x - permute(x, {0, 1, 3, 2}); }

double maxAbs(const Tensor& t) {
  double m = 0.0;
  for (const auto& x : t.d) m = std::max(m, std::abs(x));
  return m;
}


Tensor einsumN(const std::string& spec, const std::vector<const Tensor*>& ops, cplx alpha) {
  const auto arrow = spec.find("->");
  const std::string lo = spec.substr(arrow + 2);
  std::vector<std::string> labels;
  std::size_t pos = 0;
  const std::string lhs = spec.substr(0, arrow);
  while (true) {
    const auto comma = lhs.find(',', pos);
    labels.push_back(lhs.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (labels.size() != ops.size()) throw std::logic_error("UCCSD einsumN: operand count mismatch in " + spec);
  if (ops.size() == 1) {
    // relabelling, permutation and traces (repeated labels / labels missing from the output are summed)
    const Tensor& a = *ops[0];
    const std::string& la = labels[0];
    std::size_t label_dim[128] = {0};
    for (std::size_t k = 0; k < la.size(); ++k) label_dim[static_cast<int>(la[k])] = a.dims[k];
    std::vector<std::size_t> odims;
    for (char ch : lo) odims.push_back(label_dim[static_cast<int>(ch)]);
    Tensor out(odims);
    std::vector<std::size_t> ostride(lo.size(), 1);
    for (std::size_t k = lo.size(); k-- > 1;) ostride[k - 1] = ostride[k] * odims[k];
    std::vector<std::size_t> idx(la.size(), 0);
    for (std::size_t lin = 0; lin < a.size(); ++lin) {
      std::size_t rem = lin;
      for (std::size_t k = la.size(); k-- > 0;) {
        idx[k] = rem % a.dims[k];
        rem /= a.dims[k];
      }
      bool ok = true;
      std::size_t pos[128];
      for (std::size_t k = 0; k < la.size() && ok; ++k) {
        const int c = static_cast<int>(la[k]);
        bool seen = false;
        for (std::size_t m = 0; m < k; ++m)
          if (la[m] == la[k]) {
            seen = true;
            if (idx[m] != idx[k]) ok = false;
          }
        if (!seen) pos[c] = idx[k];
      }
      if (!ok) continue;
      std::size_t o_lin = 0;
      for (std::size_t k = 0; k < lo.size(); ++k) o_lin += pos[static_cast<int>(lo[k])] * ostride[k];
      out.d[o_lin] += a.d[lin];
    }
    if (alpha != cplx(1.0)) for (auto& x : out.d) x *= alpha;
    return out;
  }
  Tensor cur = *ops[0];
  std::string cur_l = labels[0];
  for (std::size_t k = 1; k < ops.size(); ++k) {
    // labels still needed after this contraction: output labels and the labels of the remaining operands
    std::string keep = lo;
    for (std::size_t r = k + 1; r < ops.size(); ++r) keep += labels[r];
    std::string out_l;
    for (char ch : cur_l + labels[k])
      if (keep.find(ch) != std::string::npos && out_l.find(ch) == std::string::npos) out_l += ch;
    const bool last = (k + 1 == ops.size());
    if (last) out_l = lo;
    cur = einsum(cur_l + "," + labels[k] + "->" + out_l, cur, *ops[k], last ? alpha : cplx(1.0));
    cur_l = out_l;
  }
  return cur;
}

Tensor takeAxis(const Tensor& a, std::size_t axis, std::size_t index) {
  std::vector<std::size_t> dims;
  for (std::size_t k = 0; k < a.dims.size(); ++k)
    if (k != axis) dims.push_back(a.dims[k]);
  Tensor out(dims);
  std::size_t outer = 1, inner = 1;
  for (std::size_t k = 0; k < axis; ++k) outer *= a.dims[k];
  for (std::size_t k = axis + 1; k < a.dims.size(); ++k) inner *= a.dims[k];
  const std::size_t n = a.dims[axis];
  for (std::size_t o = 0; o < outer; ++o)
    std::copy_n(&a.d[(o * n + index) * inner], inner, &out.d[o * inner]);
  return out;
}

void addIntoAxis(Tensor& dst, std::size_t axis, std::size_t index, const Tensor& src) {
  std::size_t outer = 1, inner = 1;
  for (std::size_t k = 0; k < axis; ++k) outer *= dst.dims[k];
  for (std::size_t k = axis + 1; k < dst.dims.size(); ++k) inner *= dst.dims[k];
  const std::size_t n = dst.dims[axis];
  if (src.size() != outer * inner) throw std::logic_error("UCCSD addIntoAxis: size mismatch");
  for (std::size_t o = 0; o < outer; ++o)
    for (std::size_t x = 0; x < inner; ++x) dst.d[(o * n + index) * inner + x] += src.d[o * inner + x];
}

}  // namespace cc
}  // namespace rerdmft
