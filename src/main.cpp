// CLI dispatch: golden|generate|tokenize|quantize|bench (flags fixed in
// the Makefile contract comments).
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  std::cerr << "heph: subcommand '" << cmd
            << "' not implemented (phase 0 stub)\n";
  return 2;
}
