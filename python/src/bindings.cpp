#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cpp/src/jakubelib.h"
#include "cpp/src/kissrandom.h"

namespace nb = nanobind;
using namespace nanobind::literals;

namespace {

template <typename ValueType>
std::vector<ValueType> to_vector(nb::handle obj, int expected_dim) {
	std::vector<ValueType> values = nb::cast<std::vector<ValueType>>(obj);
	if (expected_dim >= 0 && static_cast<int>(values.size()) != expected_dim) {
		throw nb::value_error("vector has incorrect dimensionality");
	}
	return values;
}

struct ErrorBuffer {
	char* ptr = nullptr;

	~ErrorBuffer() {
		if (ptr != nullptr) {
			std::free(ptr);
		}
	}

	std::string message(const char* fallback) const {
		return ptr != nullptr ? std::string(ptr) : std::string(fallback);
	}
};

[[noreturn]] void raise_failure(const ErrorBuffer& err, const char* fallback) {
	std::string message = err.message(fallback);
	throw nb::value_error(message.c_str());
}

template <typename IndexType, typename ValueType>
class IndexWrapper {
 public:
	explicit IndexWrapper(int dims)
			: dims_(dims), index_(std::make_unique<IndexType>(dims)) {}

	int dims() const { return dims_; }

	void add_item(int item, nb::handle vector_like) {
		auto values = to_vector<ValueType>(vector_like, dims_);
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->add_item(item, values.data(), &err.ptr);
		}
		if (!ok) {
				raise_failure(err, "add_item failed");
		}
	}

	void build(int q = 10, int n_threads = -1) {
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->build(q, n_threads, &err.ptr);
		}
		if (!ok) {
				raise_failure(err, "build failed");
		}
	}

	void unbuild() {
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->unbuild(&err.ptr);
		}
		if (!ok) {
				raise_failure(err, "unbuild failed");
		}
	}

	void unload() {
		nb::gil_scoped_release release;
		index_->unload();
	}

	void save(const std::string& path, bool prefault = false) {
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->save(path.c_str(), prefault, &err.ptr);
		}
		if (!ok) {
				raise_failure(err, "save failed");
		}
	}

	void load(const std::string& path, bool prefault = false) {
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->load(path.c_str(), prefault, &err.ptr);
		}
		if (!ok) {
				raise_failure(err, "load failed");
		}
	}

	void on_disk_build(const std::string& path) {
		ErrorBuffer err;
		bool ok;
		{
			nb::gil_scoped_release release;
			ok = index_->on_disk_build(path.c_str(), &err.ptr);
		}
		if (!ok) {
				raise_failure(err, "on_disk_build failed");
		}
	}

	void set_seed(uint64_t seed) { index_->set_seed(seed); }

	void verbose(bool value) { index_->verbose(value); }

	std::size_t n_items() const { return static_cast<std::size_t>(index_->get_n_items()); }

	std::size_t n_trees() const { return static_cast<std::size_t>(index_->get_n_trees()); }

	std::vector<ValueType> get_item(int item) const {
		std::vector<ValueType> buffer(static_cast<std::size_t>(dims_));
		{
			nb::gil_scoped_release release;
			index_->get_item(item, buffer.data());
		}
		return buffer;
	}

	ValueType get_distance(int a, int b) const {
		nb::gil_scoped_release release;
		return index_->get_distance(a, b);
	}

	nb::tuple get_nns_by_vector(nb::handle vector_like, std::size_t n, int search_k = -1) const {
		auto query = to_vector<ValueType>(vector_like, dims_);
		std::vector<int> indices;
		std::vector<ValueType> distances;
		{
			nb::gil_scoped_release release;
			index_->get_nns_by_vector(query.data(), n, search_k, &indices, &distances);
		}
		return nb::make_tuple(std::move(indices), std::move(distances));
	}

	nb::tuple get_nns_by_item(int item, std::size_t n, int search_k = -1) const {
		std::vector<int> indices;
		std::vector<ValueType> distances;
		{
			nb::gil_scoped_release release;
			index_->get_nns_by_item(item, n, search_k, &indices, &distances);
		}
		return nb::make_tuple(std::move(indices), std::move(distances));
	}

 private:
	int dims_;
	std::unique_ptr<IndexType> index_;
};

using Random64 = Jakube::Kiss64Random<uint64_t>;
using SingleThreadPolicy = Jakube::JakubeIndexSingleThreadedBuildPolicy;

using EuclideanIndex = Jakube::JakubeIndex<int, float, Jakube::Euclidean, Random64, SingleThreadPolicy>;
using AngularIndex = Jakube::JakubeIndex<int, float, Jakube::Angular, Random64, SingleThreadPolicy>;
using ManhattanIndex = Jakube::JakubeIndex<int, float, Jakube::Manhattan, Random64, SingleThreadPolicy>;
using DotProductIndex = Jakube::JakubeIndex<int, float, Jakube::DotProduct, Random64, SingleThreadPolicy>;
using HammingIndex = Jakube::JakubeIndex<int, uint64_t, Jakube::Hamming, Random64, SingleThreadPolicy>;

template <typename IndexType, typename ValueType>
void bind_index(nb::module_& m, const char* python_name, const char* metric_name) {
	const std::string doc = std::string("Jakube index using the ") + metric_name + " metric.";

	nb::class_<IndexWrapper<IndexType, ValueType>>(m, python_name, doc.c_str())
			.def(nb::init<int>(), "dims"_a, "Vector dimensionality.")
			.def("dims", &IndexWrapper<IndexType, ValueType>::dims, "Return the dimensionality.")
			.def("add_item", &IndexWrapper<IndexType, ValueType>::add_item, "item"_a, "vector"_a, "Insert a vector with the given integer id.")
			.def("build", &IndexWrapper<IndexType, ValueType>::build, "q"_a = 10, "n_threads"_a = -1, "Build the forest with q trees.")
			.def("unbuild", &IndexWrapper<IndexType, ValueType>::unbuild, "Reset the forest while keeping inserted vectors.")
			.def("unload", &IndexWrapper<IndexType, ValueType>::unload, "Release allocated resources.")
			.def("save", &IndexWrapper<IndexType, ValueType>::save, "path"_a, "prefault"_a = false, "Persist the built index to disk.")
			.def("load", &IndexWrapper<IndexType, ValueType>::load, "path"_a, "prefault"_a = false, "Load an index from disk.")
			.def("on_disk_build", &IndexWrapper<IndexType, ValueType>::on_disk_build, "path"_a, "Prepare an on-disk build target.")
			.def("set_seed", &IndexWrapper<IndexType, ValueType>::set_seed, "seed"_a, "Set the seed used during build.")
			.def("verbose", &IndexWrapper<IndexType, ValueType>::verbose, "enabled"_a, "Toggle verbose logging.")
			.def("n_items", &IndexWrapper<IndexType, ValueType>::n_items, "Return the number of stored items.")
			.def("n_trees", &IndexWrapper<IndexType, ValueType>::n_trees, "Return the number of constructed trees.")
			.def("get_item", &IndexWrapper<IndexType, ValueType>::get_item, "item"_a, "Return the stored vector for the given id.")
			.def("get_distance", &IndexWrapper<IndexType, ValueType>::get_distance, "a"_a, "b"_a, "Distance between the two stored items.")
			.def("get_nns_by_vector", &IndexWrapper<IndexType, ValueType>::get_nns_by_vector, "vector"_a, "n"_a, "search_k"_a = -1, "Return ids and distances for the nearest neighbours of the provided vector.")
			.def("get_nns_by_item", &IndexWrapper<IndexType, ValueType>::get_nns_by_item, "item"_a, "n"_a, "search_k"_a = -1, "Return ids and distances for the nearest neighbours of the stored vector.");
}

}  // namespace

NB_MODULE(jakube_ext, m) {
	m.doc() = "Python bindings for the Jakube approximate nearest neighbours library.";

	bind_index<EuclideanIndex, float>(m, "EuclideanIndex", "Euclidean");
	bind_index<AngularIndex, float>(m, "AngularIndex", "Angular");
	bind_index<ManhattanIndex, float>(m, "ManhattanIndex", "Manhattan");
	bind_index<DotProductIndex, float>(m, "DotProductIndex", "Dot product");
	bind_index<HammingIndex, uint64_t>(m, "HammingIndex", "Hamming");
}
