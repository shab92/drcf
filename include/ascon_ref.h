// ASCON-128 (Ascon v1.2) reference implementation. Plain, single-party.
// Publishers and subscribers use this with the full key K = K1 ^ K2.
#pragma once
#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace ascon {

using bytes = std::vector<uint8_t>;
constexpr uint64_t IV = 0x80400c0600000000ULL;  // k=128, r=64, a=12, b=6

inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

inline void permute(uint64_t S[5], int rounds) {
    for (int r = 12 - rounds; r < 12; r++) {
        S[2] ^= (uint64_t)(0xf0 - r * 0x10 + r);
        S[0] ^= S[4]; S[4] ^= S[3]; S[2] ^= S[1];
        uint64_t T[5];
        for (int i = 0; i < 5; i++) T[i] = ~S[i] & S[(i + 1) % 5];
        for (int i = 0; i < 5; i++) S[i] ^= T[(i + 1) % 5];
        S[1] ^= S[0]; S[0] ^= S[4]; S[3] ^= S[2]; S[2] = ~S[2];
        S[0] ^= rotr(S[0], 19) ^ rotr(S[0], 28);
        S[1] ^= rotr(S[1], 61) ^ rotr(S[1], 39);
        S[2] ^= rotr(S[2], 1) ^ rotr(S[2], 6);
        S[3] ^= rotr(S[3], 10) ^ rotr(S[3], 17);
        S[4] ^= rotr(S[4], 7) ^ rotr(S[4], 41);
    }
}

// big-endian load of up to 8 bytes (missing bytes are zero)
inline uint64_t load(const uint8_t* p, size_t n = 8) {
    uint64_t x = 0;
    for (size_t i = 0; i < 8; i++) x = (x << 8) | (i < n ? p[i] : 0);
    return x;
}
inline void store(uint8_t* p, uint64_t x, size_t n = 8) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(x >> (56 - 8 * i));
}
inline uint64_t pad_word(const uint8_t* p, size_t n) {  // n < 8 bytes + 0x80 pad
    return load(p, n) ^ (0x80ULL << (56 - 8 * n));
}

inline void init(uint64_t S[5], const uint8_t key[16], const uint8_t nonce[16], const bytes& ad) {
    uint64_t K0 = load(key), K1 = load(key + 8);
    S[0] = IV; S[1] = K0; S[2] = K1; S[3] = load(nonce); S[4] = load(nonce + 8);
    permute(S, 12);
    S[3] ^= K0; S[4] ^= K1;
    if (!ad.empty()) {
        size_t i = 0;
        for (; i + 8 <= ad.size(); i += 8) { S[0] ^= load(&ad[i]); permute(S, 6); }
        S[0] ^= pad_word(ad.data() + i, ad.size() - i); permute(S, 6);
    }
    S[4] ^= 1;
}

inline void final_tag(uint64_t S[5], const uint8_t key[16], uint8_t tag[16]) {
    uint64_t K0 = load(key), K1 = load(key + 8);
    S[1] ^= K0; S[2] ^= K1;
    permute(S, 12);
    store(tag, S[3] ^ K0); store(tag + 8, S[4] ^ K1);
}

// returns ciphertext || tag
inline bytes encrypt(const uint8_t key[16], const uint8_t nonce[16], const bytes& ad, const bytes& pt) {
    uint64_t S[5];
    init(S, key, nonce, ad);
    bytes out(pt.size() + 16);
    size_t i = 0;
    for (; i + 8 <= pt.size(); i += 8) { S[0] ^= load(&pt[i]); store(&out[i], S[0]); permute(S, 6); }
    S[0] ^= pad_word(pt.data() + i, pt.size() - i);
    store(&out[i], S[0], pt.size() - i);
    final_tag(S, key, &out[pt.size()]);
    return out;
}

inline std::optional<bytes> decrypt(const uint8_t key[16], const uint8_t nonce[16], const bytes& ad, const bytes& ct_tag) {
    if (ct_tag.size() < 16) return std::nullopt;
    size_t n = ct_tag.size() - 16;
    uint64_t S[5];
    init(S, key, nonce, ad);
    bytes pt(n);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t C = load(&ct_tag[i]); store(&pt[i], S[0] ^ C); S[0] = C; permute(S, 6); }
    size_t last = n - i;
    uint64_t C = load(&ct_tag[i], last);
    store(&pt[i], S[0] ^ C, last);
    uint64_t mask = last ? (~0ULL >> (8 * last)) : ~0ULL;
    S[0] = C ^ (S[0] & mask) ^ (0x80ULL << (56 - 8 * last));
    uint8_t tag[16];
    final_tag(S, key, tag);
    uint8_t d = 0;
    for (int j = 0; j < 16; j++) d |= tag[j] ^ ct_tag[n + j];
    if (d) return std::nullopt;
    return pt;
}

}  // namespace ascon
