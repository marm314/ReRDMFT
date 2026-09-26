#ifndef RERDMFT_UTILS_PROGRESS_H
#define RERDMFT_UTILS_PROGRESS_H

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace rerdmft {

// Live progress reporting. The program assembles most of its result report as strings and prints it only when the
// whole calculation is done, so a long run shows nothing while it works. Progress lines are therefore written, the
// moment they happen, to a separate LIVE FILE next to the input file -- `name.live` for `name.inp` -- flushed after
// every line and prefixed with the elapsed wall time:
//   [   123.4 s] (C4_DHF) FULL_OPTIMIZATION macro-iteration 3: E(total) = -113.0024..., max|g| = 1.4e-06 (NEO steps 3)
// Follow it with `tail -f name.live`. The file is truncated at the start of every run; the normal report on stdout
// is untouched. Without an open live file (unit tests, other executables) progress() does nothing.
inline std::chrono::steady_clock::time_point progressStart() {
  static const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  return start;
}

inline std::ofstream& progressFile() {
  static std::ofstream file;
  return file;
}

// Opens (truncating) the live file; returns false if it cannot be written (the run then simply has no live file).
inline bool progressOpen(const std::string& path) {
  progressFile().open(path, std::ios::out | std::ios::trunc);
  return progressFile().is_open();
}

// The method whose stage is running (NON_REL, X2C_HF, C4_DHF), shown in front of every progress line.
inline std::string& progressContext() {
  static std::string context;
  return context;
}

inline void progress(const std::string& message) {
  static std::mutex mutex;
  const std::lock_guard<std::mutex> lock(mutex);
  if (!progressFile().is_open()) return;
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - progressStart()).count();
  progressFile() << "[" << std::setw(8) << std::fixed << std::setprecision(1) << seconds << " s] "
                 << (progressContext().empty() ? "" : "(" + progressContext() + ") ") << message << std::endl;
}

// Stream-style helper: ProgressLine() << "text " << value;   (printed when the temporary ends)
class ProgressLine {
 public:
  template <typename T>
  ProgressLine& operator<<(const T& value) {
    text_ << value;
    return *this;
  }
  ~ProgressLine() { progress(text_.str()); }

 private:
  std::ostringstream text_;
};

}  // namespace rerdmft

#endif  // RERDMFT_UTILS_PROGRESS_H
