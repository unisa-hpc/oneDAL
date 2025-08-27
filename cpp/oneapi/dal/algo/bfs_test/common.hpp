#include "oneapi/dal/backend/primitives/frontier.hpp"

#include <queue>
#include <vector>
#include <map>
#include <chrono>

namespace oneapi::dal::bfs {

namespace pr = dal::backend::primitives;

class PerformanceTester {
public:
    void add(const std::string& name, const sycl::event& event) {
        if (timings_.find(name) == timings_.end()) {
            timings_[name] = std::vector<sycl::event>();
        }
        timings_[name].push_back(event);
    }

    void set_start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        round_ = start_time_;
        iteration_ = start_time_;
    }

    void round(std::string print = "") {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - round_).count();
        std::cout << " - " << print << " took: " << duration << " ms" << std::endl;
        round_ = end_time;
    }

    void iteration(int n) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - iteration_).count();
        std::cout << " - Iteration " << n << " took: " << duration << " ms" << std::endl;
        iteration_ = end_time;
    }

    void set_end() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
        std::cout << "Total execution time: " << duration << " ms" << std::endl;
    }

    void print() const {
        for (const auto& [name, events] : timings_) {
            std::cout << "Performance for " << name << ": ";
            for (const auto& event : events) {
                auto duration =
                    event.get_profiling_info<sycl::info::event_profiling::command_end>() -
                    event.get_profiling_info<sycl::info::event_profiling::command_start>();
                std::cout << duration / 1e6 << " ms ";
            }
            std::cout << std::endl;
        }
    }

private:
    // define start time
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point round_;
    std::chrono::high_resolution_clock::time_point iteration_;
    std::map<std::string, std::vector<sycl::event>> timings_;
};

void print_device_name(sycl::queue& queue) {
    const auto device = queue.get_device();
    const auto device_name = device.get_info<sycl::info::device::name>();
    std::cout << "Running on device: " << device_name << std::endl;
}

template <typename T>
std::vector<std::uint32_t> host_bfs(std::vector<T>& row_offsets,
                                    std::vector<T>& col_indices,
                                    T src) {
    std::vector<std::uint32_t> distances(row_offsets.size() - 1,
                                         std::numeric_limits<std::uint32_t>::max());
    std::queue<T> q;
    q.push(src);
    distances[src] = 0;

    while (!q.empty()) {
        T node = q.front();
        q.pop();

        for (size_t i = row_offsets[node]; i < row_offsets[node + 1]; ++i) {
            auto neighbor = col_indices[i];
            if (distances[neighbor] == std::numeric_limits<std::uint32_t>::max()) {
                distances[neighbor] = distances[node] + 1;
                q.push(neighbor);
            }
        }
    }
    return distances;
}

void test_perf(std::vector<std::uint32_t>& row_ptr,
               std::vector<std::uint32_t>& col_indices,
               std::uint32_t src = 0) {
    sycl::queue queue =
        sycl::queue(sycl::default_selector_v, sycl::property::queue::enable_profiling());
    print_device_name(queue);

    PerformanceTester perf_tester;

    std::vector<std::uint32_t> weights(col_indices.size(), 1);

    auto graph = pr::csr_graph(queue, row_ptr, col_indices, weights);
    size_t num_nodes = row_ptr.size() - 1;
    auto in_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    auto out_frontier = pr::frontier<std::uint32_t>(queue, num_nodes, sycl::usm::alloc::device);
    pr::ndarray<std::uint32_t, 1> distance =
        pr::ndarray<std::uint32_t, 1>::empty(queue,
                                             { static_cast<std::int64_t>(num_nodes) },
                                             sycl::usm::alloc::device);
    auto distance_ptr = distance.get_mutable_data();

    queue
        .submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(num_nodes), [=](sycl::id<1> idx) {
                distance_ptr[idx] = (idx[0] == src ? 0 : num_nodes + 1);
            });
        })
        .wait_and_throw();

    in_frontier.insert(src);
    size_t iter = 0;

    /// Start BFS
    perf_tester.set_start();
    while (!in_frontier.empty()) {
        std::cout << "Iteration " << iter << std::endl;
        perf_tester.round("empty");
        auto e = pr::advance(graph,
                             in_frontier,
                             out_frontier,
                             [=](auto vertex, auto neighbor, auto edge, auto weight) {
                                 bool visited = distance_ptr[neighbor] < num_nodes + 1;
                                 if (!visited) {
                                     distance_ptr[neighbor] = iter + 1;
                                 }
                                 return !visited;
                             });
        e.wait_and_throw();
        perf_tester.round("advance");
        auto duration = e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                        e.get_profiling_info<sycl::info::event_profiling::command_start>();
        std::cout << "Advance Kernel Time: " << (duration / 1e6) << " ms\n";
        iter++;
        pr::swap_frontiers(in_frontier, out_frontier);
        perf_tester.round("swap_frontiers");
        out_frontier.clear();
        perf_tester.round("clear");
        perf_tester.iteration(iter);
    }
    /// End BFS
    perf_tester.set_end();
}

} // namespace oneapi::dal::bfs
