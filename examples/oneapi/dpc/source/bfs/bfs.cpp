#include <sycl/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/bfs_test.hpp"
#include <tuple>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <iostream>

namespace dal = oneapi::dal::bfs;

struct DummyCSR {
    std::uint32_t num_vertices = 0;
    std::uint32_t num_edges = 0; // number of stored edges (after symmetrization if any)
    std::vector<std::uint32_t> row_ptr; // size = num_vertices + 1
    std::vector<std::uint32_t> col_indices; // size = num_edges
};

// Load an edge list from a text file and build CSR (0-based, contiguous vertex ids).
// File format: each line contains two integers: src dst. Lines starting with '#' are ignored.
// If undirected=true, edges are symmetrized (both (u,v) and (v,u) are inserted).
static DummyCSR load_edge_list_as_csr(const std::string &path, bool undirected = true) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open edge list file: " + path);
    }

    // Read and parse the first line (header: "#V #V #E")
    std::string line;
    if (!std::getline(fin, line)) {
        fin.close();
        return DummyCSR{}; // empty file
    }
    std::istringstream header_iss(line);
    std::uint64_t num_v1 = 0, num_v2 = 0, num_e = 0;
    header_iss >> num_v1 >> num_v2 >> num_e;
    // Ignore the header values after parsing

    // Read edges and collect unique vertex ids
    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    edges.reserve(1 << 20); // pre-reserve some space to reduce reallocations
    std::vector<std::uint32_t> verts;
    verts.reserve(1 << 20);

    std::uint64_t line_no = 1;
    while (std::getline(fin, line)) {
        ++line_no;
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream iss(line);
        std::int64_t u64, v64; // allow for negative after decrement
        if (!(iss >> u64 >> v64)) {
            // Skip malformed lines silently; you can also throw if preferred
            continue;
        }
        if (u64 < 0 || v64 < 0) {
            // skip edges with negative indices after decrement
            continue;
        }

        // Stash raw edge
        edges.emplace_back(static_cast<std::uint32_t>(u64), static_cast<std::uint32_t>(v64));
        verts.push_back(static_cast<std::uint32_t>(u64));
        verts.push_back(static_cast<std::uint32_t>(v64));
    }
    fin.close();

    if (edges.empty()) {
        return DummyCSR{}; // empty graph
    }

    // Build a compact 0..n-1 mapping for arbitrary vertex ids
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const std::uint32_t n = static_cast<std::uint32_t>(verts.size());

    std::unordered_map<std::uint32_t, std::uint32_t> id2comp;
    id2comp.reserve(verts.size() * 1.3);
    for (std::uint32_t i = 0; i < n; ++i)
        id2comp[verts[i]] = i;

    // Count degrees (with optional symmetrization)
    std::vector<std::uint32_t> degree(n, 0);
    auto bump = [&](std::uint32_t a) {
        if (a < n)
            ++degree[a];
    };

    for (const auto &e : edges) {
        std::uint32_t u = id2comp[e.first];
        std::uint32_t v = id2comp[e.second];
        bump(u);
        if (undirected)
            bump(v);
    }

    // Prefix sum -> row_ptr
    DummyCSR csr;
    csr.num_vertices = n;
    csr.row_ptr.resize(n + 1, 0);
    for (std::uint32_t i = 0; i < n; ++i)
        csr.row_ptr[i + 1] = csr.row_ptr[i] + degree[i];
    csr.num_edges = csr.row_ptr[n];
    csr.col_indices.resize(csr.num_edges);

    // Temporary write positions per row
    std::vector<std::uint32_t> next_pos = csr.row_ptr;

    // Fill adjacency
    for (const auto &e : edges) {
        std::uint32_t u = id2comp[e.first];
        std::uint32_t v = id2comp[e.second];
        csr.col_indices[next_pos[u]++] = v;
        if (undirected)
            csr.col_indices[next_pos[v]++] = u;
    }

    // Sort neighbors within each row for deterministic traversal
    for (std::uint32_t i = 0; i < n; ++i) {
        auto begin = csr.row_ptr[i];
        auto end = csr.row_ptr[i + 1];
        std::sort(csr.col_indices.begin() + begin, csr.col_indices.begin() + end);
    }

    return csr;
}

/*
// Example usage (kept disabled to avoid interfering with existing tests):
try {
  DummyCSR g = load_edge_list_as_csr("edges.txt", /*undirected=*\/true);
  std::cout << "V=" << g.num_vertices << ", E=" << g.num_edges << "\n";
} catch (const std::exception &e) {
  std::cerr << "Error: " << e.what() << "\n";
}
*/

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <edge_list_file>\n";
        return 1;
    }
    const std::string edge_list_file = argv[1];

    DummyCSR graph;
    graph = load_edge_list_as_csr(edge_list_file, /*undirected=*/true);
    std::cout << "Row Pointer: ";
    for (int i = 0; i < 20; i++) {
        std::cout << graph.row_ptr[i] << " ";
    }
    std::cout << std::endl << "Column Indices: ";
    for (int i = 0; i < 20; i++) {
        std::cout << graph.col_indices[i] << " ";
    }
    std::cout << std::endl;

    dal::test_perf(graph.row_ptr, graph.col_indices, /*src=*/3);
}