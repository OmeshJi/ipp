#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

void bfs_parallel(const std::vector<std::vector<int>>& graph, int start) {
    int n = graph.size();
    std::vector<bool> visited(n, false);
    std::vector<int> current_frontier = {start};
    visited[start] = true;

    while (!current_frontier.empty()) {
        std::vector<int> next_frontier;

        // Parallelize over current frontier
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < current_frontier.size(); ++i) {
            int node = current_frontier[i];
            std::cout << "Visited: " << node << " (thread " << omp_get_thread_num() << ")\n";

            for (int neighbor : graph[node]) {
                bool should_add = false;

                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        should_add = true;
                    }
                }

                if (should_add) {
                    #pragma omp critical
                    next_frontier.push_back(neighbor);
                }
            }
        }

        current_frontier = next_frontier;
    }
}

int main() {
    // Example graph: undirected graph with 6 nodes (0 to 5)
    std::vector<std::vector<int>> graph = {
        {1, 2},    // Neighbors of node 0
        {0, 3, 4}, // Neighbors of node 1
        {0, 4},    // Neighbors of node 2
        {1, 5},    // Neighbors of node 3
        {1, 2, 5}, // Neighbors of node 4
        {3, 4}     // Neighbors of node 5
    };

    int start_node = 0;

    std::cout << "Starting parallel BFS from node " << start_node << "...\n";
    bfs_parallel(graph, start_node);

    return 0;
}

