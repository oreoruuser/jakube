#pragma once

#include <cstdint>
#include <cstddef>

namespace Jakube{
    template <typename seed_t>
    class Kiss64Random {
        uint64_t x, y, z, c;
        uint64_t default_seed = 1234567890987654321ULL;
    public:
        Kiss64Random (seed_t seed):
            x{seed},
            y{362436362436362436ULL},
            z{1066149217761810ULL},
            c{123456123456123456ULL}
            {}
        Kiss64Random () : 
            Kiss64Random(default_seed) {}
        auto kiss() ;
        auto flip();
        size_t index(seed_t n);
        void set_seed(seed_t seed);
    };
}
template <typename seed_t>
auto Jakube::Kiss64Random<seed_t>::kiss(){
    z = 6906969069LL*z+1234567;

    // Xor shift
    y ^= (y<<13);
    y ^= (y>>17);
    y ^= (y<<43);

    // Multiply-with-carry (uint128_t t = (2^58 + 1) * x + c; c = t >> 64; x = (uint64_t) t)
    auto t = (x<<58)+c;
    c = (x>>6);
    x += t;
    c += (x<t);

    return x + y + z;
}
template <typename seed_t>
auto Jakube::Kiss64Random<seed_t>::kiss(){
    return kiss() & 1;
}
template <typename seed_t>
size_t  Jakube::Kiss64Random<seed_t>::index(seed_t n){
    if (n == 0) return 0;
    return kiss() % n;
}
template <typename seed_t>
void Jakube::Kiss64Random<seed_t>::set_seed(seed_t seed){
    x = seed;
}