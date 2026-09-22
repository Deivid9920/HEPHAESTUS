// Self-contained SHA-256 (FIPS 180-4) for artifact integrity: the
// loader verifies the weights file digest against the manifest and can
// expose per-tensor digests. No third-party dependencies.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace heph {

// Returns the lowercase hex digest of len bytes at data.
std::string sha256_hex(const uint8_t* data, size_t len);

}  // namespace heph
