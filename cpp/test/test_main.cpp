/*
 * COMPILE WITH:
 * g++ -std=c++17 -O2 -o test_runner test_jakube.cpp -pthread
 *
 * (You might need -msse, -mavx, etc. if you force-enable them,
 * but the code should auto-detect)
 */

// This define tells doctest to create a main() function
#include "doctest.h"

// Your library headers
#include "kissrandom.h"
#include "jakubelib.h"

// Standard library includes for testing
#include <vector>
#include <cmath>
#include <cstdio> // For std::remove (file cleanup)
#include <cstdint>

// --- Helper Aliases ---
// We'll use the Kiss64Random you provided as the RNG
using MyRandom = Jakube::Kiss64Random<uint64_t>;
// And test with the single-threaded build policy for simplicity
using SingleThreadPolicy = Jakube::JakubeIndexSingleThreadedBuildPolicy;

// Create type aliases for each metric
using EuclideanIndex = Jakube::JakubeIndex<int, float, Jakube::Euclidean, MyRandom, SingleThreadPolicy>;
using AngularIndex = Jakube::JakubeIndex<int, float, Jakube::Angular, MyRandom, SingleThreadPolicy>;
using ManhattanIndex = Jakube::JakubeIndex<int, float, Jakube::Manhattan, MyRandom, SingleThreadPolicy>;
using DotProductIndex = Jakube::JakubeIndex<int, float, Jakube::DotProduct, MyRandom, SingleThreadPolicy>;
using HammingIndex = Jakube::JakubeIndex<int, uint64_t, Jakube::Hamming, MyRandom, SingleThreadPolicy>;


// --- Tests Start Here ---

TEST_CASE("Metric Structs - Direct Distance Checks") {
    // Test the metric structs' static distance functions directly
    int f = 3;

    SUBCASE("Euclidean Distance") {
        using Node = Jakube::Euclidean::Node<int, float>;
        Node n1, n2;
        float v1[] = {1.0f, 2.0f, 3.0f};
        float v2[] = {4.0f, 5.0f, 6.0f};
        memcpy(n1.v, v1, f * sizeof(float));
        memcpy(n2.v, v2, f * sizeof(float));
        
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = (-3)^2 + (-3)^2 + (-3)^2 = 9 + 9 + 9 = 27
        CHECK(Jakube::Euclidean::distance(&n1, &n2, f) == doctest::Approx(27.0));
        // Normalized: sqrt(27)
        CHECK(Jakube::Euclidean::normalized_distance(27.0) == doctest::Approx(sqrt(27.0)));
    }

    SUBCASE("Angular Distance") {
        using Node = Jakube::Angular::Node<int, float>;
        Node n1, n2, n3;
        float v1[] = {1.0f, 0.0f, 0.0f}; // x-axis
        float v2[] = {0.0f, 1.0f, 0.0f}; // y-axis (90 deg)
        float v3[] = {2.0f, 0.0f, 0.0f}; // x-axis (0 deg, diff norm)
        memcpy(n1.v, v1, f * sizeof(float));
        memcpy(n2.v, v2, f * sizeof(float));
        memcpy(n3.v, v3, f * sizeof(float));
        
        // Must init nodes to calculate norm
        Jakube::Angular::init_node(&n1, f); // norm = 1
        Jakube::Angular::init_node(&n2, f); // norm = 1
        Jakube::Angular::init_node(&n3, f); // norm = 4

        // cos(theta) = (v1 . v2) / (|v1| |v2|) = 0 / (1*1) = 0
        // dist = 2 - 2*cos(theta) = 2.0
        CHECK(Jakube::Angular::distance(&n1, &n2, f) == doctest::Approx(2.0));
        
        // cos(theta) = (v1 . v3) / (|v1|*|v3|) = 2 / (1*2) = 1
        // dist = 2 - 2*cos(theta) = 0.0
        CHECK(Jakube::Angular::distance(&n1, &n3, f) == doctest::Approx(0.0));
        
        // Normalized: sqrt(2.0)
        CHECK(Jakube::Angular::normalized_distance(2.0) == doctest::Approx(sqrt(2.0)));
    }

    SUBCASE("Hamming Distance") {
        using Node = Jakube::Hamming::Node<int, uint64_t>;
        Node n1, n2;
        // f=1, T=uint64_t
        // 1010... (binary) = 0xAAAAAAAAAAAAAAAA
        // 1111... (binary) = 0xFFFFFFFFFFFFFFFF
        // XOR = 0101... (binary) = 0x5555555555555555
        // popcount(0x55...) = 32
        n1.v[0] = 0xAAAAAAAAAAAAAAAAULL;
        n2.v[0] = 0xFFFFFFFFFFFFFFFFULL;
        CHECK(Jakube::Hamming::distance(&n1, &n2, 1) == 32);
        
        n1.v[0] = 0x1; // ...0001
        n2.v[0] = 0x3; // ...0011
        // XOR = 0x2 (...0010), popcount = 1
        CHECK(Jakube::Hamming::distance(&n1, &n2, 1) == 1);
    }
}

TEST_CASE("JakubeIndex - Euclidean Workflow") {
    int f = 2;
    EuclideanIndex index(f);

    std::vector<float> v0 = {1.0f, 1.0f};
    std::vector<float> v1 = {2.0f, 2.0f};
    std::vector<float> v2 = {10.0f, 10.0f};
    std::vector<float> v3 = {1.0f, 2.0f};
    
    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());
    index.add_item(3, v3.data());

    CHECK(index.get_n_items() == 4);

    // Test get_item
    std::vector<float> v_out(f);
    index.get_item(3, v_out.data());
    CHECK(v_out[0] == doctest::Approx(1.0f));
    CHECK(v_out[1] == doctest::Approx(2.0f));

    // Test get_distance (pre-build)
    // dist = sqrt((1-2)^2 + (1-2)^2) = sqrt(1+1) = sqrt(2)
    CHECK(index.get_distance(0, 1) == doctest::Approx(sqrt(2.0)));
    // dist = sqrt((1-10)^2 + (1-10)^2) = sqrt(81+81) = sqrt(162)
    CHECK(index.get_distance(0, 2) == doctest::Approx(sqrt(162.0)));

    // Build
    index.build(10); // 10 trees
    CHECK(index.get_n_trees() == 10);

    // Test get_nns_by_vector
    std::vector<float> q = {0.0f, 0.0f};
    std::vector<int> result;
    std::vector<float> distances;
    index.get_nns_by_vector(q.data(), 4, -1, &result, &distances);

    REQUIRE(result.size() == 4);
    REQUIRE(distances.size() == 4);
    
    // Expected order: 0, 3, 1, 2
    CHECK(result[0] == 0); // {1,1} -> dist = sqrt(2)
    CHECK(result[1] == 3); // {1,2} -> dist = sqrt(5)
    CHECK(result[2] == 1); // {2,2} -> dist = sqrt(8)
    CHECK(result[3] == 2); // {10,10} -> dist = sqrt(200)

    CHECK(distances[0] == doctest::Approx(sqrt(2.0)));
    CHECK(distances[1] == doctest::Approx(sqrt(5.0)));
    CHECK(distances[2] == doctest::Approx(sqrt(8.0)));
    CHECK(distances[3] == doctest::Approx(sqrt(200.0)));

    // Test get_nns_by_item (find items near item 0: {1,1})
    result.clear();
    distances.clear();
    index.get_nns_by_item(0, 4, -1, &result, &distances);

    REQUIRE(result.size() == 4);
    // Expected order: 0, 3, 1, 2
    // 0: {1,1} -> self, dist = 0
    // 3: {1,2} -> dist = sqrt((1-1)^2 + (1-2)^2) = 1
    // 1: {2,2} -> dist = sqrt((1-2)^2 + (1-2)^2) = sqrt(2)
    // 2: {10,10} -> dist = sqrt((1-10)^2 + (1-10)^2) = sqrt(162)
    CHECK(result[0] == 0);
    CHECK(result[1] == 3);
    CHECK(result[2] == 1);
    CHECK(result[3] == 2);
    
    CHECK(distances[0] == doctest::Approx(0.0));
    CHECK(distances[1] == doctest::Approx(1.0));
    CHECK(distances[2] == doctest::Approx(sqrt(2.0)));
    CHECK(distances[3] == doctest::Approx(sqrt(162.0)));
}

TEST_CASE("JakubeIndex - DotProduct Workflow") {
    // DotProduct is special: it uses preprocess and postprocess
    // and its distance definition changes after build.
    int f = 2;
    DotProductIndex index(f);

    std::vector<float> v0 = {1.0f, 1.0f};
    std::vector<float> v1 = {2.0f, 0.0f};
    std::vector<float> v2 = {-1.0f, 0.0f};

    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());
    
    // Build (this triggers preprocess and postprocess)
    index.build(10);
    
    // After build, get_distance should return NEGATIVE DOT PRODUCT
    CHECK(index.get_distance(0, 1) == doctest::Approx(-2.0)); // -(1*2 + 1*0)
    CHECK(index.get_distance(0, 2) == doctest::Approx(1.0));  // -(1*-1 + 1*0)
    CHECK(index.get_distance(1, 2) == doctest::Approx(2.0));  // -(2*-1 + 0*0)

    // Test query q = {3.0, 0.0}
    std::vector<float> q = {3.0f, 0.0f};
    std::vector<int> result;
    std::vector<float> distances;
    index.get_nns_by_vector(q.data(), 3, -1, &result, &distances);

    // Expected order by dot product: 1, 0, 2
    // Dot(q, v1) = 6
    // Dot(q, v0) = 3
    // Dot(q, v2) = -3
    // Distances are negative dot product
    
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 1);
    CHECK(result[1] == 0);
    CHECK(result[2] == 2);

    REQUIRE(distances.size() == 3);
    CHECK(distances[0] == doctest::Approx(-6.0));
    CHECK(distances[1] == doctest::Approx(-3.0));
    CHECK(distances[2] == doctest::Approx(3.0));
}

TEST_CASE("JakubeIndex - Hamming Workflow") {
    int f = 1; // 1 uint64_t
    HammingIndex index(f);
    
    std::vector<uint64_t> v0 = {0x1ULL}; // ...0001
    std::vector<uint64_t> v1 = {0x3ULL}; // ...0011
    std::vector<uint64_t> v2 = {0xFULL}; // ...1111

    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());

    // Test get_distance
    CHECK(index.get_distance(0, 1) == 1); // popcount(0x1 ^ 0x3 = 0x2)
    CHECK(index.get_distance(0, 2) == 3); // popcount(0x1 ^ 0xF = 0xE)
    CHECK(index.get_distance(1, 2) == 2); // popcount(0x3 ^ 0xF = 0xC)
    
    index.build(10);

    std::vector<uint64_t> q = {0x0ULL}; // ...0000
    std::vector<int> result;
    std::vector<uint64_t> distances;
    index.get_nns_by_vector(q.data(), 3, -1, &result, &distances);

    // Expected order: 0, 1, 2
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 0);
    CHECK(result[1] == 1);
    CHECK(result[2] == 2);

    REQUIRE(distances.size() == 3);
    CHECK(distances[0] == 1); // popcount(0^1)
    CHECK(distances[1] == 2); // popcount(0^3)
    CHECK(distances[2] == 4); // popcount(0^F)
}


TEST_CASE("JakubeIndex - Save/Load Workflow") {
    int f = 3;
    const char* filename = "test_index.jakube";

    // 1. Create, build, and save
    {
        AngularIndex index_to_save(f);
        std::vector<float> v0 = {1.0f, 0.0f, 0.0f};
        std::vector<float> v1 = {0.0f, 1.0f, 0.0f};
        index_to_save.add_item(0, v0.data());
        index_to_save.add_item(1, v1.data());
        index_to_save.build(5);
        
        CHECK(index_to_save.get_n_items() == 2);
        CHECK(index_to_save.save(filename) == true);
    } // index_to_save goes out of scope, unloading its data

    // 2. Load and test
    {
        AngularIndex index_to_load(f);
        char* error = NULL;
        CHECK(index_to_load.load(filename, false, &error) == true);
        if (error) free(error);

        CHECK(index_to_load.get_n_items() == 2);
        CHECK(index_to_load.get_n_trees() == 5);
        
        // Test get_distance (should be sqrt(2))
        CHECK(index_to_load.get_distance(0, 1) == doctest::Approx(sqrt(2.0)));

        // Test query
        std::vector<float> q = {0.9f, 0.1f, 0.0f}; // close to v0
        std::vector<int> result;
        index_to_load.get_nns_by_vector(q.data(), 2, -1, &result, NULL);
        
        REQUIRE(result.size() == 2);
        CHECK(result[0] == 0); // v0 should be closer
        CHECK(result[1] == 1);

        index_to_load.unload();
    }

    // 3. Cleanup
    std::remove(filename);
}

