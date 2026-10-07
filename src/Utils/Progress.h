#ifndef RERDMFT_UTILS_PROGRESS_H
#define RERDMFT_UTILS_PROGRESS_H

#include <chrono>
#include <cmath>
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
// every line and prefixed with the elapsed wall time, broken into weeks/days/hours/minutes/seconds (only the units
// actually needed, e.g. a sub-minute run just shows seconds; once a larger unit appears, every smaller one down to
// seconds is shown zero-padded):
//   [            3.4s] (C4_DHF) ...
//   [         2m03.4s] (C4_DHF) ...
//   [      1h02m03.4s] (C4_DHF) ...
//   [   2d01h02m03.4s] (C4_DHF) FULL_OPTIMIZATION macro-iteration 3: E(total) = -113.0024..., max|g| = 1.4e-06 (NEO steps 3)
//   [1w02d01h02m03.4s] (C4_DHF) ...
// Follow it with `tail -f name.live`. The file is truncated at the start of every run; the normal report on stdout
// is untouched. Without an open live file (unit tests, other executables) progress() does nothing.
inline std::chrono::steady_clock::time_point progressStart() {
  static const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  return start;
}

// Formats `seconds` as "[Nw][NNd][NNh][NNm]N.Ns" -- the largest nonzero unit is shown unpadded, every
// unit from there down to seconds is then shown zero-padded (even if zero) so columns stay aligned
// once a run crosses an hour/day/week boundary; below that, just the plain "N.Ns" seconds field.
inline std::string formatElapsedWdhms(double seconds) {
  if (seconds < 0.0) seconds = 0.0;
  // Round to the displayed precision (0.1s) FIRST, then decompose into units -- otherwise a value
  // like 59.95 decomposes as "0m 59.95s", which itself then rounds to "59.9" or "60.0" at print
  // time, the latter reading like "60.0s" when it should have rolled over to "1m00.0s" instead.
  seconds = std::round(seconds * 10.0) / 10.0;
  long long whole = static_cast<long long>(seconds);
  const double frac = seconds - static_cast<double>(whole);
  const long long w = whole / 604800;
  whole %= 604800;
  const long long d = whole / 86400;
  whole %= 86400;
  const long long h = whole / 3600;
  whole %= 3600;
  const long long m = whole / 60;
  whole %= 60;
  const double s = static_cast<double>(whole) + frac;

  std::ostringstream out;
  bool started = false;
  if (w > 0) {
    out << w << "w";
    started = true;
  }
  if (started) out << std::setw(2) << std::setfill('0') << d << "d";
  else if (d > 0) {
    out << d << "d";
    started = true;
  }
  if (started) out << std::setw(2) << std::setfill('0') << h << "h";
  else if (h > 0) {
    out << h << "h";
    started = true;
  }
  if (started) out << std::setw(2) << std::setfill('0') << m << "m";
  else if (m > 0) {
    out << m << "m";
    started = true;
  }
  if (started) out << std::setw(4) << std::setfill('0');
  out << std::fixed << std::setprecision(1) << s << "s";
  return out.str();
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
  progressFile() << "[" << std::setw(17) << std::setfill(' ') << formatElapsedWdhms(seconds) << "] "
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
