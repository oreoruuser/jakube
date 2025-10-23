#pragma once

template<typename T = unsigned long long>
class kissrand {
    T x,y,z,c;
public:
    kissrand(T seed = 1234567890987654321ULL) {
        x = seed;
        y = 362436362436362436ULL;
        z = 1066149217761810ULL;
        c = 123456123456123456ULL;
    }

    T kiss() {
        z = 6906969069ULL * z + 1234567;
        y ^= (y << 13); 
        y ^= (y >> 17);
        y ^= (y << 43);

        T t = (x << 58) + c;
        c = (x >> 6);
        x += t;
        c += (x < t);

        return x + y + z;
    }

    int flip() { 
        return kiss() & 1; 
    }
    int index(int n) { 
        return kiss() % n; 
    }
    void set_seed(T seed) { 
        x = seed; 
    }
};
