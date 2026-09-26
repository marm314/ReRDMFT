#ifndef RERDMFT_UTILS_BLASTHREADS_H
#define RERDMFT_UTILS_BLASTHREADS_H

#include <dlfcn.h>

namespace rerdmft {

// The OpenBLAS pthread build starts its own thread pool for every BLAS call, so a BLAS call made from inside an
// OpenMP parallel region oversubscribes the machine (n_omp x n_blas threads) and can make small products 10-15x
// slower than running them serially -- measured on the slab integral transform. SerialBlasScope makes BLAS
// single-threaded for the duration of a parallel region that calls it (the region supplies the parallelism) and
// restores the previous setting afterwards. The OpenBLAS control functions are looked up at run time (dlsym) so the
// code links and works with any BLAS: with one that lacks them (or an OpenMP build of OpenBLAS, which already runs
// serially inside parallel regions) the guard simply does nothing.
class SerialBlasScope {
 public:
  SerialBlasScope() {
    get_ = reinterpret_cast<GetFn>(dlsym(RTLD_DEFAULT, "openblas_get_num_threads"));
    set_ = reinterpret_cast<SetFn>(dlsym(RTLD_DEFAULT, "openblas_set_num_threads"));
    if (get_ != nullptr && set_ != nullptr) {
      saved_ = get_();
      set_(1);
    }
  }
  ~SerialBlasScope() {
    if (saved_ > 0 && set_ != nullptr) set_(saved_);
  }
  SerialBlasScope(const SerialBlasScope&) = delete;
  SerialBlasScope& operator=(const SerialBlasScope&) = delete;

 private:
  using GetFn = int (*)();
  using SetFn = void (*)(int);
  GetFn get_ = nullptr;
  SetFn set_ = nullptr;
  int saved_ = -1;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_BLASTHREADS_H
