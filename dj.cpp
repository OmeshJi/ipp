#include <iostream>
#include <omp.h>
#include <climits> // For INT_MAX

using namespace std;

#define V 9
#define INF INT_MAX

void dijkstra_openmp(int graph[V][V], int src) {
    int dist[V];
    bool visited[V];

    // Initialize distances and visited status
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        visited[i] = false;
    }
    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = -1;
        int min_val = INF;

        // Parallel block to find vertex with minimum distance
        #pragma omp parallel
        {
            int local_u = -1;
            int local_min = INF;

            #pragma omp for nowait
            for (int v = 0; v < V; v++) {
                if (!visited[v] && dist[v] < local_min) {
                    local_min = dist[v];
                    local_u = v;
                }
            }

            #pragma omp critical
            {
                if (local_min < min_val) {
                    min_val = local_min;
                    u = local_u;
                }
            }
        }

        if (u == -1) break; // No reachable vertex

        visited[u] = true;

        // Parallel block to relax neighbors
        #pragma omp parallel for
        for (int v = 0; v < V; v++) {
            if (!visited[v] && graph[u][v] && dist[u] != INF &&
                dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }

    cout << "Vertex \t Distance from Source\n";
    for (int i = 0; i < V; i++) {
        cout << i << "\t\t";
        if (dist[i] == INF)
            cout << "INF\n";
        else
            cout << dist[i] << "\n";
    }
}

int main() {
    int graph[V][V] = {
        {0, 4, 0, 0, 0, 0, 0, 8, 0},
        {4, 0, 8, 0, 0, 0, 0, 11, 0},
        {0, 8, 0, 7, 0, 4, 0, 0, 2},
        {0, 0, 7, 0, 9, 14, 0, 0, 0},
        {0, 0, 0, 9, 0, 10, 0, 0, 0},
        {0, 0, 4, 14, 10, 0, 2, 0, 0},
        {0, 0, 0, 0, 0, 2, 0, 1, 6},
        {8, 11, 0, 0, 0, 0, 1, 0, 7},
        {0, 0, 2, 0, 0, 0, 6, 7, 0}
    };

    int source = 0;
    dijkstra_openmp(graph, source);

    return 0;
}

