#ifndef RERDMFT_UTILS_OUTPUTFORMAT_H
#define RERDMFT_UTILS_OUTPUTFORMAT_H

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <streambuf>
#include <string>

namespace rerdmft {

// Section title: a blank line, the title and an underline of the same length.
// The leading '\x01' is consumed by SectionIndentBuf (it marks the title); a stream without that filter would show it.
inline void printSectionTitle(std::ostream& out, const std::string& title) {
  out << "\n\x01" << title << "\n" << std::string(title.size(), '=') << "\n";
}

// One energy line with the decimal points of every such line in the report on the same column: `indent` spaces, the label padded
// so that the value's right edge is always at column indent + label_width + value_width, 10 fixed decimals. SCF totals (indent 2)
// and the component blocks below them (indent 4) therefore line up.
inline void printEnergyLine(std::ostream& out, std::size_t indent, const std::string& label, double value,
                            const std::string& unit = " Hartree") {
  const std::ios_base::fmtflags flags = out.flags();
  const std::streamsize precision = out.precision();
  out << std::string(indent, ' ') << std::setw(static_cast<int>(37 - indent)) << std::left << (label + ":") << std::right
      << std::fixed << std::setprecision(10) << std::setw(20) << value << unit << "\n";
  out.flags(flags);
  out.precision(precision);
}

// A small number (energy change, residual, gradient) in scientific notation, right-aligned so that the mantissa dots line up.
inline void printScientificLine(std::ostream& out, std::size_t indent, const std::string& label, double value,
                                const std::string& unit = "") {
  const std::ios_base::fmtflags flags = out.flags();
  const std::streamsize precision = out.precision();
  out << std::string(indent, ' ') << std::setw(static_cast<int>(37 - indent)) << std::left << (label + ":") << std::right
      << std::scientific << std::setprecision(3) << std::setw(20) << value << unit << "\n";
  out.flags(flags);
  out.precision(precision);
}

// h-onebody / Vee / N-N, the breakdown printed after every final energy.
inline void printEnergyComponents(std::ostream& out, std::size_t indent, double h_onebody, double vee, double nuclear_repulsion) {
  printEnergyLine(out, indent, "h-onebody", h_onebody);
  printEnergyLine(out, indent, "Vee", vee);
  printEnergyLine(out, indent, "N-N", nuclear_repulsion);
}

// std::cout filter that orders the report: below a section title every block gets the same layout. A "block" starts at a non-blank
// line, after a blank line (or the title underline), with indentation <= 2; the block's lines are shifted so that its first line sits
// at column 2, keeping their relative indentation. Before the first title and after a '\x02' marker (the farewell art) lines pass
// untouched; '\x01' (see printSectionTitle) marks a title line.
class SectionIndentBuf : public std::streambuf {
 public:
  explicit SectionIndentBuf(std::streambuf* sink) : sink_(sink) {}
  ~SectionIndentBuf() override { if (!line_.empty()) emit(line_, false); }

 protected:
  int_type overflow(int_type c) override {
    if (c == traits_type::eof()) return traits_type::not_eof(c);
    if (c == '\n') {
      emit(line_, true);
      line_.clear();
    } else if (c == '\x02' && line_.empty()) {
      enabled_ = false;
    } else {
      line_.push_back(static_cast<char>(c));
    }
    return c;
  }
  int sync() override { return sink_->pubsync(); }

 private:
  void emit(std::string line, bool newline) {
    if (!line.empty() && line[0] == '\x01') {  // title
      line.erase(0, 1);
      enabled_ = true;
      delta_ = 0;
      underline_next_ = true;
      prev_blank_ = true;
    } else if (enabled_) {
      const std::size_t first = line.find_first_not_of(' ');
      if (underline_next_) {
        underline_next_ = false;
      } else if (first == std::string::npos) {
        line.clear();
        prev_blank_ = true;
      } else {
        if (prev_blank_ && first <= 2) delta_ = 2 - first;
        prev_blank_ = false;
        line.insert(0, delta_, ' ');
      }
    }
    sink_->sputn(line.data(), static_cast<std::streamsize>(line.size()));
    if (newline) sink_->sputc('\n');
  }
  std::streambuf* sink_;
  std::string line_;
  bool enabled_ = false;
  bool prev_blank_ = true;
  bool underline_next_ = false;
  std::size_t delta_ = 0;
};

// std::cout filter: wraps every line longer than `width` columns at a space (continuation lines keep the line's indentation plus
// two spaces); a line without a usable space is broken hard. Lines up to `width` pass through untouched.
class LineWrapBuf : public std::streambuf {
 public:
  LineWrapBuf(std::streambuf* sink, std::size_t width) : sink_(sink), width_(width) {}
  ~LineWrapBuf() override { flushLine(false); }

 protected:
  int_type overflow(int_type c) override {
    if (c == traits_type::eof()) return traits_type::not_eof(c);
    if (c == '\n') {
      flushLine(true);
    } else {
      line_.push_back(static_cast<char>(c));
    }
    return c;
  }
  int sync() override { return sink_->pubsync(); }

 private:
  void flushLine(bool newline) {
    std::string rest = line_;
    line_.clear();
    const std::size_t indent = rest.find_first_not_of(' ') == std::string::npos ? 0 : rest.find_first_not_of(' ');
    bool first = true;
    while (rest.size() > width_) {
      std::size_t cut = rest.rfind(' ', width_);
      const std::size_t floor = first ? indent + 1 : indent + 3;
      if (cut == std::string::npos || cut <= floor) cut = width_;
      sink_->sputn(rest.data(), static_cast<std::streamsize>(cut));
      sink_->sputc('\n');
      std::size_t next = cut;
      while (next < rest.size() && rest[next] == ' ') ++next;
      rest = std::string(indent + 2, ' ') + rest.substr(next);
      first = false;
    }
    sink_->sputn(rest.data(), static_cast<std::streamsize>(rest.size()));
    if (newline) sink_->sputc('\n');
  }
  std::streambuf* sink_;
  std::size_t width_;
  std::string line_;
};

}  // namespace rerdmft

#endif
