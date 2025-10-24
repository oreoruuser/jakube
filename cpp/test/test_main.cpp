#include "doctest.h"
#include "kissrandom.h"

#include <map>       // For statistical tests
#include <numeric>   // For std::accumulate
#include <cmath>     // For std::fabs
#include <set>       // For bit distribution test

// --- Behavior & Determinism Tests (Template-based) ---

// Test against a wide variety of unsigned integer types.
TEST_CASE_TEMPLATE("Kiss64Random Instantiation",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    // Alias for the specific Kiss64Random instantiation under test.
    using TestRandom = Jakube::Kiss64Random<TestType>;

    SUBCASE("Default constructor") {
        // Verifies default constructor does not throw.
        CHECK_NOTHROW(TestRandom rand_default);
    }

    SUBCASE("Seeded constructor") {
        // Verifies seeded constructor does not throw.
        TestType seed = static_cast<TestType>(12345);
        CHECK_NOTHROW(TestRandom rand_seeded(seed));
    }
}

TEST_CASE_TEMPLATE("Kiss64Random Determinism (Default Seed)",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    using TestRandom = Jakube::Kiss64Random<TestType>;

    TestRandom rand1;
    TestRandom rand2;

    // Default-constructed generators must produce identical sequences.
    for(int i = 0; i < 10; ++i) {
        CHECK(rand1.kiss() == rand2.kiss());
    }
}

TEST_CASE_TEMPLATE("Kiss64Random Determinism (Explicit Seed)",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed = static_cast<TestType>(42);

    TestRandom rand1(seed);
    TestRandom rand2(seed);

    // Generators with the same seed must produce identical sequences.
    for(int i = 0; i < 10; ++i) {
        CHECK(rand1.kiss() == rand2.kiss());
    }
}

TEST_CASE_TEMPLATE("Kiss64Random State (Different Seeds)",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed1 = static_cast<TestType>(100);
    TestType seed2 = static_cast<TestType>(200);

    TestRandom rand1(seed1);
    TestRandom rand2(seed2);

    // Different seeds should produce different sequences. (Low prob. of collision)
    CHECK_NE(rand1.kiss(), rand2.kiss());
    CHECK_NE(rand1.kiss(), rand2.kiss());
}


TEST_CASE_TEMPLATE("Kiss64Random `set_seed` Resets State",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestType seed1 = static_cast<TestType>(100);
    // *** THIS IS THE LINE THAT WAS FIXED ***
    TestType seed2 = static_cast<TestType>(200);

    TestRandom rand1(seed1);
    TestRandom rand2(seed2);

    // Generate a sequence from the first generator.
    auto val1_a = rand1.kiss();
    auto val1_b = rand1.kiss();
    auto val1_c = rand1.kiss();

    // Verify generators are initially in different states.
    CHECK_NE(val1_a, rand2.kiss());

    // Reset the second generator to the first's seed.
    rand2.set_seed(seed1);

    // Verify the second generator's sequence now matches the first.
    CHECK(rand2.kiss() == val1_a);
    CHECK(rand2.kiss() == val1_b);
    CHECK(rand2.kiss() == val1_c);
}

TEST_CASE_TEMPLATE("Kiss64Random Helper Functions",
                   TestType,
                   std::uint64_t,
                   std::uint32_t,
                   std::uint16_t,
                   unsigned int,
                   unsigned long,
                   unsigned long long) {
    using TestRandom = Jakube::Kiss64Random<TestType>;
    TestRandom rand(static_cast<TestType>(999));

    SUBCASE("flip() range") {
        for (int i = 0; i < 100; ++i) {
            auto f = rand.flip();
            // Result must be 0 or 1.
            CHECK((f == 0 || f == 1));
        }
    }

    SUBCASE("index(n) range") {
        std::size_t n = 50;
        for (int i = 0; 200 > i; ++i) {
            auto idx = rand.index(n);
            // Verifies range [0, n-1]. (idx >= 0 is implicit for unsigned).
            CHECK(idx < n);
        }
    }

    SUBCASE("index(1) behavior") {
        for (int i = 0; 10 > i; ++i) {
            // index(1) must always return 0.
            CHECK(rand.index(1) == 0);
        }
    }

    SUBCASE("index(0) behavior") {
        // Verify index(0) behavior (expected to return 0).
        CHECK(rand.index(0) == 0);
    }
}


// --- Statistical Sanity Checks (uint64_t specific) ---
// These tests are hard-coded for the uint64_t generator as they
// check the properties of the full 64-bit output.

TEST_CASE("Kiss64Random<uint64_t> statistical properties") {
    using TestRandom = Jakube::Kiss64Random<std::uint64_t>;
    TestRandom rand(12345ULL);

    SUBCASE("flip() distribution") {
        int heads = 0;
        int N = 10000;
        for (int i = 0; i < N; ++i) {
            if (rand.flip() == 1) {
                heads++;
            }
        }
        double ratio = static_cast<double>(heads) / N;
        // Check for a statistically reasonable deviation from the 0.5 mean.
        CHECK(ratio > 0.48);
        CHECK(ratio < 0.52);
    }

    SUBCASE("index(10) uniform distribution") {
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

        double expected_count = static_cast<double>(N) / n;
        double expected_mean = (n - 1) / 2.0;
        double actual_mean = sum / N;

        // Verify the mean is close to the expected value.
        CHECK(std::fabs(actual_mean - expected_mean) < 0.05);

        // Verify each bucket's count is within a reasonable tolerance.
        for (auto const& [key, count] : counts) {
            CHECK(count > expected_count * 0.95);
            CHECK(count < expected_count * 1.05);
        }
    }

    SUBCASE("kiss() high bit variance") {
        std::set<std::uint64_t> high_bytes;
        int N = 1000;
        for(int i = 0; i < N; ++i) {
            // Isolate the top byte of the 64-bit output.
            high_bytes.insert(rand.kiss() >> 56);
        }
        
        // Sanity check that high bits are being utilized.
        // Check for a reasonable distribution of high-byte values.
        CHECK(high_bytes.size() > 256 / 4);
    }
}
