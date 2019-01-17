#if !defined(BFS_H)
#define      BFS_H

#include <stddef.h>
#include <stdint.h>

/*
 * This defines `cse6230graph` to be an *opaque pointer* to an unspecified
 * struct.  In your implementation, you should define this struct to hold
 * whatever data you want to define a graph type.  Because the routines in
 * this interface do not *dereference* the pointer (directly access the data
 * members), we will be able to use the same interface for everyone's
 * implementation.
 */
typedef struct _cse6230graph * cse6230graph;

/** Create a graph structure.

  \param[in] num_edges    the number of input edges
  \param[in] edges        the input edges as pairs of vertices:
                            - the graph is undirected, so the ordering of the vertices is irrelevant
                            - the same edge may be specified multiple times
                            - self-edges may be ignored
  \param[out] graph_p     a pointer to the initialized graph structure
 */
int graph_create(size_t num_edges, const int64_t (* edges)[2], cse6230graph *graph_p);

/** Conduct a breadth first search on a graph structure.

  \param[in] graph        a graph as created in graph_create()
  \param[in] num_keys     the size of \a key
  \param[in] key          an array of keys, which are vertices in the graph
  \param[in/out] parents  parents[i] is an array for key[i]: it has already
                          been allocated but must be filled in the following manner:
                            - a key is its own parent, parents[i][key[i]] = key[i]
                            - a vertex \a j that cannot be reached by breadth-first search from key[i] has parent -1,
                              parents[i][j] = -1
                            - otherwise, the parent should be a valid parent in a breadth-first search, so
                              parents[i][j] should be a vertex that is closer to key[i] than j.
 */
int breadth_first_search(cse6230graph graph, int num_keys, const int64_t *key, int64_t **parents);
/** Destroy a graph structure.
  \param[in/out] graph_p  A pointer to the graph created in graph_create().
                          All data associated with the graph will be deallocated.
 */
int graph_destroy(cse6230graph *graph_p);
#endif
