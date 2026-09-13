#ifndef RERDMFT_SHELL_H
#define RERDMFT_SHELL_H

#include <vector>

namespace rerdmft {

// A contracted shell of primitive Gaussians sharing an angular momentum.
// Coefficients are stored exactly as given in the basis set file (no
// normalization applied).
struct Shell {
  int l = 0;  // angular momentum: 0=S, 1=P, 2=D, 3=F, 4=G, ...
  std::vector<double> exponents;
  std::vector<double> coefficients;
};

// Cartesian angular momentum exponents (lx, ly, lz), with lx+ly+lz == l.
struct CartesianExponents {
  int lx = 0;
  int ly = 0;
  int lz = 0;
};

// Enumerates the cartesian components of a shell with angular momentum l, in
// the standard order (S: 1; P: x,y,z; D: xx,xy,xz,yy,yz,zz; ...).
std::vector<CartesianExponents> cartesianComponents(int l);

// Conventional single-letter label for a shell's angular momentum
// (S, P, D, F, G, H, I, ...); '?' if l is out of the supported range.
char angularMomentumLabel(int l);

}  // namespace rerdmft

#endif  // RERDMFT_SHELL_H
