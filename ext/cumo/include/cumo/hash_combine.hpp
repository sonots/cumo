#ifndef CUMO_HASH_COMBINE_H
#define CUMO_HASH_COMBINE_H

#include <cstddef>

namespace cumo {
namespace internal {

// Borrowed from boost::hash_combine
//
// TODO(sonots): hash combine in 64bit
inline void HashCombine(std::size_t& seed, std::size_t hash_value) { seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

}  // namespace internal
}  // namespace cumo

#endif /* ifndef CUMO_HASH_COMBINE_H */
