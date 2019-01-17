#include <stdlib.h>
#include "bfs.h"

struct _cse6230graph
{
    int n_vertices;
    int **adj_matrix;
};


int graph_create(size_t num_edges, const int64_t (*edges)[2], cse6230graph *graph)
{
    cse6230graph g = NULL;

    g = (cse6230graph) malloc(sizeof(*g));
    if (!g) return 1;

    // find edge with highest ID
    int64_t biggest = 0;
    for (int e = 0; e < num_edges; e++) {
        if (edges[e][0] > biggest) {biggest = edges[e][0];}
        if (edges[e][1] > biggest) {biggest = edges[e][1];}
        }

    int N = biggest + 1;
    g->n_vertices = N;

    // allocate memory for adjacency matrix
    g->adj_matrix = (int**) malloc(N * sizeof(int*));
    for (int n = 0; n < N; n++) {
        g->adj_matrix[n] = (int*) malloc(N * sizeof(int));
    }

    // initialize adjacency matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            g->adj_matrix[i][j] = -1;
        }
    }

    // fill out adjacency matrix
    for (int i = 0; i < num_edges; i++){
        int64_t u = edges[i][0];
        int64_t v = edges[i][1];
        g->adj_matrix[u][v] = 1;
        g->adj_matrix[v][u] = 1;
    }

    *graph = g;
    return 0;
}

int graph_destroy(cse6230graph *graph)
{
    for (int i = 0; i < (*graph)->n_vertices; i++) {
        free((*graph)->adj_matrix[i]);
    }
    free((*graph)->adj_matrix);
    free(*graph);
    *graph = NULL;
    return 0;
}

int breadth_first_search(cse6230graph graph, int num_keys, const int64_t *keys, int64_t **parents)
{
    int N = graph->n_vertices;

    // initialize parents array
    for (int i = 0; i < num_keys; i++) {
        for (int v = 0; v < N; v++) {
            parents[i][v] = -1;
        }
    }

    // one BFS traversal per key
    for (int k = 0; k < num_keys; k++) {

        int queue[N];
        int head = 0;
        int tail = 1;
        int key = keys[k];
        int x;

	    parents[k][key] = key;

        // initialize visited array and queue for each key's traversal
        for (int c = 0; c < N; c++) {
            queue[c] = -2;
        }

        // breadth-first search from key
        queue[head] = key;
        for (head = 0; head < N; head++) {
	           if (head >= tail) {
	                  break;
	    }
            x = queue[head];
            // iterate through row x in adj matrix
            for (int j = 0; j < N; j++) {
                // check if edge (x,j) exists
                if (graph->adj_matrix[x][j] == 1) {
                    // check if j has been visited
                    int seen = 0;
		            if (parents[k][j] != -1) {
			        seen = 1;
			        }

                    // if not visited, fill in parents, add child to queue
                    if (seen == 0) {
                        parents[k][j] = x;
                        queue[tail] = j;
                        tail++;
                    }
                }
            }
        }
    }
    return 0;
}
