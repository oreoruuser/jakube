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

#include <map>       // For statistical tests
#include <numeric>   // For std::accumulate
#include <cmath>     // For std::fabs
#include <set>       // For bit distribution test
#include <algorithm> // For std::sort

// Standard library includes for testing
#include <vector>
#include <cmath>
#include <cstdio> // For std::remove (file cleanup)
#include <cstdint>

// --- Helper Aliases ---
// We'll use the Kiss64Random you provided as the RNG
using MyRandom = Jakube::Kiss64Random<uint64_t>;

// Build policies
using SingleThreadPolicy = Jakube::JakubeIndexSingleThreadedBuildPolicy;
using MultiThreadPolicy = Jakube::JakubeIndexMultiThreadedBuildPolicy;

// Create type aliases for each metric
using EuclideanIndex = Jakube::JakubeIndex<int, float, Jakube::Euclidean, MyRandom, SingleThreadPolicy>;
using AngularIndex = Jakube::JakubeIndex<int, float, Jakube::Angular, MyRandom, SingleThreadPolicy>;
using ManhattanIndex = Jakube::JakubeIndex<int, float, Jakube::Manhattan, MyRandom, SingleThreadPolicy>;
using DotProductIndex = Jakube::JakubeIndex<int, float, Jakube::DotProduct, MyRandom, SingleThreadPolicy>;
using HammingIndex = Jakube::JakubeIndex<int, uint64_t, Jakube::Hamming, MyRandom, SingleThreadPolicy>;

// Aliases for new tests
using EuclideanIndexMulti = Jakube::JakubeIndex<int, float, Jakube::Euclidean, MyRandom, MultiThreadPolicy>;
using EuclideanIndexDouble = Jakube::JakubeIndex<int, double, Jakube::Euclidean, MyRandom, SingleThreadPolicy>;


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

TEST_CASE("JakubeIndex - Euclidean Workflow") { // euclidean_index_test
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

TEST_CASE("JakubeIndex - Angular Workflow") { // angular_index_test
    int f = 2;
    AngularIndex index(f);

    std::vector<float> v0 = {1.0f, 0.0f}; // x-axis
    std::vector<float> v1 = {0.0f, 1.0f}; // y-axis
    std::vector<float> v2 = {-1.0f, 0.0f}; // -x-axis
    
    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());

    index.build(10);
    
    // Test get_distance (dist = sqrt(2 - 2*cos(theta)))
    CHECK(index.get_distance(0, 1) == doctest::Approx(sqrt(2.0))); // 90 deg
    CHECK(index.get_distance(0, 2) == doctest::Approx(2.0)); // 180 deg
    CHECK(index.get_distance(1, 2) == doctest::Approx(sqrt(2.0))); // 90 deg

    // Query vector slightly off the x-axis
    std::vector<float> q = {0.9f, 0.1f};
    std::vector<int> result;
    std::vector<float> distances;
    index.get_nns_by_vector(q.data(), 3, -1, &result, &distances);

    // Expected order: 0, 1, 2
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 0);
    CHECK(result[1] == 1);
    CHECK(result[2] == 2);
}

TEST_CASE("JakubeIndex - Manhattan Workflow") { // manhattan_index_test
    int f = 2;
    ManhattanIndex index(f);

    std::vector<float> v0 = {1.0f, 1.0f};
    std::vector<float> v1 = {2.0f, 2.0f};
    std::vector<float> v2 = {10.0f, 10.0f};
    std::vector<float> v3 = {1.0f, 2.0f};
    
    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    index.add_item(2, v2.data());
    index.add_item(3, v3.data());

    index.build(10);

    // Test get_distance (dist = |x1-x2| + |y1-y2|)
    CHECK(index.get_distance(0, 1) == doctest::Approx(2.0)); // |1-2| + |1-2|
    CHECK(index.get_distance(0, 3) == doctest::Approx(1.0)); // |1-1| + |1-2|
    CHECK(index.get_distance(0, 2) == doctest::Approx(18.0)); // |1-10| + |1-10|

    // Test query
    std::vector<float> q = {0.0f, 0.0f};
    std::vector<int> result;
    std::vector<float> distances;
    index.get_nns_by_vector(q.data(), 4, -1, &result, &distances);

    // Expected order: 0, 3, 1, 2
    REQUIRE(result.size() == 4);
    CHECK(result[0] == 0); // dist = 2
    CHECK(result[1] == 3); // dist = 3
    CHECK(result[2] == 1); // dist = 4
    CHECK(result[3] == 2); // dist = 20
    
    CHECK(distances[0] == doctest::Approx(2.0));
    CHECK(distances[1] == doctest::Approx(3.0));
    CHECK(distances[2] == doctest::Approx(4.0));
    CHECK(distances[3] == doctest::Approx(20.0));
}

TEST_CASE("JakubeIndex - DotProduct Workflow") { // dot_index_test
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

TEST_CASE("JakubeIndex - Hamming Workflow") { // hamming_index_test
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


TEST_CASE("JakubeIndex - On-Disk Save/Load Workflow") { // on_disk_build_test
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

TEST_CASE("JakubeIndex - Index with Holes") { // holes_test
    EuclideanIndex index(2);
    
    std::vector<float> v0 = {1.0f, 1.0f};
    std::vector<float> v5 = {5.0f, 5.0f};
    std::vector<float> v10 = {10.0f, 10.0f};

    // Add items with non-contiguous indices
    index.add_item(0, v0.data());
    index.add_item(5, v5.data());
    index.add_item(10, v10.data());

    // Library keeps the highest seen index + 1, so holes increase the count.
    CHECK(index.get_n_items() == 11);
    index.build(10);
    
    // Check that get_item works with holes
    std::vector<float> v_out(2);
    index.get_item(5, v_out.data());
    CHECK(v_out[0] == doctest::Approx(5.0f));
    
    // Check that querying works
    std::vector<float> q = {0.0f, 0.0f};
    std::vector<int> result;
    index.get_nns_by_vector(q.data(), 3, -1, &result, NULL);
    
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 0);   // dist sqrt(2)
    CHECK(result[1] == 5);   // dist sqrt(50)
    CHECK(result[2] == 10);  // dist sqrt(200)
}

TEST_CASE("JakubeIndex - Multithreaded Build") { // multithreaded_build_test
    int f = 2;
    EuclideanIndexMulti index(f);
    
    std::vector<float> mt0 = {1.0f, 1.0f};
    std::vector<float> mt1 = {2.0f, 2.0f};
    std::vector<float> mt2 = {10.0f, 10.0f};
    index.add_item(0, mt0.data());
    index.add_item(1, mt1.data());
    index.add_item(2, mt2.data());
    
    // Build with multiple threads (e.g., 4)
    // The number of threads is an optional argument
    index.build(10, 4); 
    
    CHECK(index.get_n_trees() == 10);
    
    std::vector<float> q = {0, 0};
    std::vector<int> result;
    index.get_nns_by_vector(q.data(), 3, -1, &result, NULL);
    
    // Results should be the same as the single-threaded build
    REQUIRE(result.size() == 3);
    CHECK(result[0] == 0);
    CHECK(result[1] == 1);
    CHECK(result[2] == 2);
}

TEST_CASE("JakubeIndex - Generic Functionality (Edge Cases)") { // index_test
    EuclideanIndex index(2);
    std::vector<float> initial = {1.0f, 1.0f};
    index.add_item(0, initial.data());
    CHECK(index.get_n_items() == 1);
    
    index.build(10);
    
    // Test query after build
    std::vector<int> result;
    index.get_nns_by_item(0, 1, -1, &result, NULL);
    CHECK(result[0] == 0);
    
    // Test unload
    index.unload();
    CHECK(index.get_n_items() == 0);
    CHECK(index.get_n_trees() == 0);
    
    // Test re-use (reinitialization)
    std::vector<float> reuse0 = {10.0f, 10.0f};
    std::vector<float> reuse1 = {11.0f, 11.0f};
        EuclideanIndex reused(2);
        reused.add_item(0, reuse0.data());
        reused.add_item(1, reuse1.data());
        CHECK(reused.get_n_items() == 2);
        reused.build(5);
        CHECK(reused.get_n_trees() == 5);

        std::vector<float> q = {9, 9};
        result.clear();
        reused.get_nns_by_vector(q.data(), 1, -1, &result, NULL);
        CHECK(result[0] == 0);
}

TEST_CASE("JakubeIndex - Different Data Types") { // types_test
    int f = 2;
    EuclideanIndexDouble index(f);
    
    std::vector<double> v0 = {1.0, 1.0};
    std::vector<double> v1 = {2.0, 2.0};
    index.add_item(0, v0.data());
    index.add_item(1, v1.data());
    
    index.build(10);
    
    // Test distance with double
    CHECK(index.get_distance(0, 1) == doctest::Approx(sqrt(2.0)));
    
    std::vector<double> q = {0.0, 0.0};
    std::vector<int> result;
    index.get_nns_by_vector(q.data(), 2, -1, &result, NULL);
    
    REQUIRE(result.size() == 2);
    CHECK(result[0] == 0);
    CHECK(result[1] == 1);
}

TEST_CASE("JakubeIndex - Build Determinism (Seeding)") { // seed_test
    int f = 10;
    std::vector<int> results1, results2;
    std::vector<float> q(f);
    for(int z=0; z<f; z++) q[z] = (float)z;
    
    std::vector<std::vector<float>> items;
    for (int i = 0; i < 100; i++) {
        std::vector<float> v(f);
        for(int z=0; z<f; z++) v[z] = (float)i + (z*0.1f);
        items.push_back(v);
    }

    // Build 1
    {
        EuclideanIndex index1(f);
        index1.set_seed(123); // Set seed
        for (int i = 0; i < 100; i++) {
            index1.add_item(i, items[i].data());
        }
        index1.build(10);
        index1.get_nns_by_vector(q.data(), 10, -1, &results1, NULL);
    }
    
    // Build 2
    {
        EuclideanIndex index2(f);
        index2.set_seed(123); // Set same seed
        for (int i = 0; i < 100; i++) {
            index2.add_item(i, items[i].data());
        }
        index2.build(10);
        index2.get_nns_by_vector(q.data(), 10, -1, &results2, NULL);
    }

    // Results should be identical due to same seed
    CHECK(results1 == results2);
}

TEST_CASE("JakubeIndex - Accuracy (Recall)") { // accuracy_test
    int f = 10;
    int n_items = 1000;
    EuclideanIndex index(f);
    MyRandom rand(42);

    std::vector<std::vector<float>> items(n_items, std::vector<float>(f));
    for (int i = 0; i < n_items; i++) {
        for (int j = 0; j < f; j++) {
            items[i][j] = (float)rand.kiss() / (float)UINT64_MAX;
        }
        index.add_item(i, items[i].data());
    }
    index.build(20); // Build 20 trees

    // Get ground truth for 10 queries
    int n_queries = 10;
    int k = 10;
    int correct_hits = 0;

    for (int i = 0; i < n_queries; i++) {
        std::vector<float> q = items[i]; // Use an existing item as query
        
        // Get ground truth
        std::vector<std::pair<float, int>> ground_truth;
        for (int j = 0; j < n_items; j++) {
            ground_truth.push_back({index.get_distance(i, j), j});
        }
        std::sort(ground_truth.begin(), ground_truth.end());

        // Get Jakube results
        std::vector<int> result;
        // Search more nodes (e.g., n_trees * k * 5) for better accuracy
        index.get_nns_by_vector(q.data(), k, k * 20 * 5, &result, NULL);

        // Check recall
        std::set<int> truth_set;
        for (int j = 0; j < k; j++) {
            truth_set.insert(ground_truth[j].second);
        }
        
        for (int res_item : result) {
            if (truth_set.count(res_item)) {
                correct_hits++;
            }
        }
    }
    
    double recall = (double)correct_hits / (n_queries * k);
    // Recall should be high (e.g., >90%)
    CHECK(recall > 0.9);
}

TEST_CASE("JakubeIndex - Memory Leak (Placeholder)") { // memory_leak_test
    SUBCASE("Build/Unload Loop") {
        // This test case is a placeholder.
        // True memory leak detection requires tools like Valgrind,
        // AddressSanitizer (ASan), or LeakSanitizer (LSan).
        // Running this test executable under such a tool would reveal leaks.
        for (int i = 0; i < 10; i++) {
            EuclideanIndex* index = new EuclideanIndex(10);
            for (int j = 0; j < 100; j++) {
                std::vector<float> v(10, (float)j);
                index->add_item(j, v.data());
            }
            index->build(10);
            // index->unload(); // Not unloading would be a leak
            delete index; // This should free all memory
        }
        CHECK(true); // If it runs without crashing, it passes here.
    }
}


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