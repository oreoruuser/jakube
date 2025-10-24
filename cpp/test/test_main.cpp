#include "doctest.h"
#include "kissrandom.h" // Your header file (which includes <cstdint>)

#include <map>       // For statistical tests
#include <numeric>   // For std::accumulate
#include <cmath>     // For std::fabs
#include <set>       // For bit distribution test

// The 'using TestRandom' alias must be *inside* each template test case
// because 'TestType' is only defined there.
TEST_CASE_TEMPLATE("Kiss64Random can be instantiated", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;

    // Test default constructor
    CHECK_NOTHROW(TestRandom rand_default);

    // Test seeded constructor
    TestType seed = static_cast<TestType>(12345);
    CHECK_NOTHROW(TestRandom rand_seeded(seed));
}

TEST_CASE_TEMPLATE("Kiss64Random default constructor is deterministic", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;

    TestRandom rand1;
    TestRandom rand2;

    // With no seed, the sequence of numbers must be identical
    for(int i = 0; i < 10; ++i) {
        CHECK(rand1.kiss() == rand2.kiss());
    }
}

TEST_CASE_TEMPLATE("Kiss64Random is deterministic with the same seed", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed = static_cast<TestType>(42);

    TestRandom rand1(seed);
    TestRandom rand2(seed);

    // With the same seed, the sequence of numbers must be identical
    for(int i = 0; i < 10; ++i) {
        CHECK(rand1.kiss() == rand2.kiss());
    }
}

TEST_CASE_TEMPLATE("Kiss64Random produces different sequences with different seeds", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed1 = static_cast<TestType>(100);
    TestType seed2 = static_cast<TestType>(200);

    TestRandom rand1(seed1);
    TestRandom rand2(seed2);

    // It's technically possible for a collision, but overwhelmingly
    // unlikely on the first few calls.
    CHECK_NE(rand1.kiss(), rand2.kiss());
    CHECK_NE(rand1.kiss(), rand2.kiss());
}


TEST_CASE_TEMPLATE("Kiss64Random `set_seed` resets the generator state", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed1 = static_cast<TestType>(100);
    TestType seed2 = static_cast<TestType>(200);

    TestRandom rand1(seed1);
    TestRandom rand2(seed2);

    // Get some initial values from rand1
    auto val1_a = rand1.kiss();
    auto val1_b = rand1.kiss();
    auto val1_c = rand1.kiss();

    // Sanity check: they should be different
    CHECK_NE(val1_a, rand2.kiss());

    // Now, reset rand2 to rand1's original seed
    rand2.set_seed(seed1);

    // The sequence from rand2 should now exactly match rand1's original sequence
    CHECK(rand2.kiss() == val1_a);
    CHECK(rand2.kiss() == val1_b);
    CHECK(rand2.kiss() == val1_c);
}

TEST_CASE_TEMPLATE("Kiss64Random helper functions (flip, index)", TestType, std::uint64_t) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestRandom rand(static_cast<TestType>(999));

    SUBCASE("flip() returns 0 or 1") {
        for (int i = 0; i < 100; ++i) {
            auto f = rand.flip();
            // Check that the result is always 0 or 1
            CHECK((f == 0 || f == 1));
        }
    }

    SUBCASE("index(n) returns a value in the correct range [0, n-1]") {
        std::size_t n = 50;
        for (int i = 0; 200 > i; ++i) {
            auto idx = rand.index(n);
            CHECK(idx < n); // This also checks idx >= 0 since it's unsigned
        }
    }

    SUBCASE("index(1) always returns 0") {
        // Because `x % 1` is always 0
        for (int i = 0; 10 > i; ++i) {
            CHECK(rand.index(1) == 0);
        }
    }

    SUBCASE("index(0) returns 0") {
        // Your code has a specific check for n == 0
        CHECK(rand.index(0) == 0);
    }
}


// --- Statistical Sanity Checks ---
// These are not template tests, so the alias can be here.
TEST_CASE("Kiss64Random<uint64_t> simple statistical properties") {
    using TestRandom = Jakube::Kiss64Random<std::uint64_t>;
    TestRandom rand(12345ULL);

    SUBCASE("flip() has a roughly 50/50 distribution") {
        int heads = 0;
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            if (rand.flip() == 1) {
                heads++;
            }
        }
        double ratio = static_cast<double>(heads) / N;
        // Check for a reasonable deviation from 0.5
        // (For N=10000, 99.7% of results should be within 0.5 +/- 0.015)
        CHECK(ratio > 0.48);
        CHECK(ratio < 0.52);
    }

    SUBCASE("index(10) has a roughly uniform distribution") {
        std::size_t n = 10;
        int N = 100000;
        std::map<std::size_t, int> counts;
        double sum = 0;
        for (std::size_t i = 0; i < n; ++i) {
            counts[i] = 0;
        }

        for (int i = 0; i < N; ++i) {
            auto idx = rand.index(n);
            counts[idx]++;
            sum += idx;
        }

        double expected_count = static_cast<double>(N) / n; // 10,000
        double expected_mean = (n - 1) / 2.0; // 4.5
        double actual_mean = sum / N;

        // Check that the mean is close to the expected mean
        CHECK(std::fabs(actual_mean - expected_mean) < 0.05);

        // Check that each bucket is within a reasonable percentage of the expected count
        for (auto const& [key, count] : counts) {
            CHECK(count > expected_count * 0.95);
            CHECK(count < expected_count * 1.05);
        }
    }

    SUBCASE("kiss() output has high bit variance") {
        // This is a simple test to check that the high bits aren't just 0.
        // It's not a true test of randomness, but catches simple mistakes.
        std::set<std::uint64_t> high_bytes;
        int N = 1000;
        for(int i = 0; i < N; ++i) {
            // Extract the top byte
            high_bytes.insert(rand.kiss() >> 56);
        }
        
        // Check that we got a good variety of values in the top byte
        // e.g., more than 1/4 of all possible 256 values
        CHECK(high_bytes.size() > 256 / 4);
    }
}
