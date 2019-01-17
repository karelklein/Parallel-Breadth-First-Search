#if !defined(BFS_H)
#define      BFS_H

#include <stddef.h>
#include <mpi.h>

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct _bfsgraph *BFSGraph;

/** Create a breadth first search context object
 *
 * This object encapsulates all the decisions about how best to implement BFS
 * for various number of processors and problem sizes
 *
 * comm: input, MPI communicator describing the parallel layout.
 * out:  graph_p, the object that will define the graph on which we will
 * search.
 */
int BFSGraphCreate(MPI_Comm comm, BFSGraph *graph_p);
/** Destroy the object */
int BFSGraphDestroy(BFSGraph *graph_p);

/** Get arrays for edges to be filled.
  \param[in]     E             global number of edges that will be in the graph
  \param[out]    numEdgesLocal the number of edges you would like to assign to the current MPI process.
  \param[out]    eLocal_p      This should be made to point to an array, allocated by you, where you would like the local edges to be stored.
 */
int BFSGraphGetEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**eLocal_p)[2]);

/** Restore arrays allocated for edges
  \param[in]     E             global number of edges that will be in the graph
  \param[in]     numEdgesLocal the number of edges you would like to assign to the current MPI process.
  \param[in/out] eLocal_p      deallocate this array where local edges are stored.
 */
int BFSGraphRestoreEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**eLocal_p)[2]);

/** Set edges of the graph / set up the graph data structure

  \param[in]     E             global number of edges in the graph
  \param[in]     numEdgesLocal the number of edges assigned to the current MPI process.
  \param[in]     eLocal_p      the local edges.  you should make your own copy of these edges, because they may be freed before the graph is done being used.
  */
int BFSGraphSetEdges(BFSGraph graph, size_t E, size_t numEdgesLocal, const int64_t (*eLocal_p)[2]);

/** After the graph has been set up, get arrays for the parents of vertices.

  \param[out]    numVerticesLocal  the number of vertices you are assigning to this MPI process
  \param[out]    firstVertexLocal  the first vertex you are assigning to this
                                   MPI process (so this process will return parents for vertices
                                   [firstLocalVertex, ..., firstLocalVertex + numVerticesLocal -1]).
  \param[out]    parentsLocal      should be made to point to an allocated array where the local parents can be written.
 */
int BFSGraphGetParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal);

/** Restore arrays for the parents of the vertices.

  \param[in]    numVerticesLocal  the number of vertices you are assigning to this MPI process
  \param[in]    firstVertexLocal  the first vertex you are assigning to this
                                  MPI process (so this process will return parents for vertices
                                  [firstLocalVertex, ..., firstLocalVertex + numVerticesLocal -1]).
  \param[in/out]    parentsLocal  should deallocate the array of parents
 */
int BFSGraphRestoreParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal);

/** Create the parent list associated with a lists of keys.

  \param[in]    num_keys          the number of search keys
  \param[in]    key               a list of the keys (duplicated on every MPI process)
  \param[in]    numVerticesLocal  the number of vertices in the local portion of the output parents array
  \param[in]    firstVertexLocal  the index of the first vertex assigned to this process
  \param[in/out] parentsLocal     parentsLocal[i] is an array for key[i]: it has already
                                  been allocated but must be filled in the following manner:
                                    - a key is its own parent: if firstVertexLocal <= key[i] < firstVertexLocal + numVerticesLocal, then
                                      parentsLocal[i][key[i] - firstVertexLocal] = key[i]
                                    - a vertex j in [firstVertexLocal, firstVertexLocal + numVerticesLocal) that cannot be reached by breadth-first search from key[i] has parent -1,
                                      parents[i][j - firstVertexLocal] = -1
                                    - otherwise, the parent should be a valid parent in a breadth-first search, so
                                      parents[i][j - firstVertexLocal] should be a vertex that is closer to key[i] than j.
 */
int BFSGraphSearch(BFSGraph graph, int num_keys, const int64_t *key, size_t numVerticesLocal, int64_t firstLocalVertex, int64_t **parentsLocal);

// ----------------------------------------- cuda implementation -------------------------------------------------------- //

/** Get arrays for edges to be filled.
  \param[in]     E             global number of edges that will be in the graph
  \param[in]     numDevices    the number of cuda devices available
  \param[out]    numEdgesLocal the number of edges you would like to assign to each cuda device
  \param[out]    eLocal_p      For each device, this should be made to point to an array, allocated *on the device* by you, where you would like the local edges to be stored;
                               This can be NULL for any device that is assigned 0 edges.
 */

int BFSGraphGetEdgeArraysDevice(BFSGraph graph, size_t E, int numDevices, size_t numEdgesLocal[], int64_t (*eLocal_p[])[2]);

/** Restore arrays for edges to be filled.
  \param[in]     E             global number of edges that will be in the graph
  \param[in]     numDevices    the number of cuda devices available
  \param[in]     numEdgesLocal the number of edges you would like to assign to each cuda device
  \param[in/out] eLocal_p      For each device, deallocate the array for edges that was allocated above.
 */
int BFSGraphRestoreEdgeArraysDevice(BFSGraph graph, size_t E, int numDevices, size_t numEdgesLocal[], int64_t (*eLocal_p[])[2]);

/** Set edges of the graph / set up the graph data structure

  \param[in]     E             global number of edges in the graph
  \param[in]     numDevices    the number of cuda devices available
  \param[in]     numEdgesLocal the number of edges assigned to each device
  \param[in]     eLocal_p      the local edges for each device.  you should make your own copy of these edges, because they may be freed before the graph is done being used.
  */
int BFSGraphSetEdgesDevice(BFSGraph graph, size_t E, int numDevices, const size_t numEdgesLocal[], const int64_t (*eLocal_p[])[2]);

/** After the graph has been set up, get arrays for the parents of vertices.

  \param[in]     numDevices        the number of cuda devices available
  \param[out]    numVerticesLocal  the number of vertices you are assigning to each device
  \param[out]    firstVertexLocal  the first vertex you are assigning to each cuda device
                                   (so device i will have an array for the parents of the vertices
                                   [firstLocalVertex[i], ..., firstLocalVertex[i] + numVerticesLocal[i] -1]).
  \param[out]    parentsLocal      should be made to point to an array allocated *on the device memory* where the local parents can be written.
                                   This can be NULL for any device that is assigned 0 edges.
 */
int BFSGraphGetParentArraysDevice(BFSGraph graph, int numDevices, size_t numVerticesLocal[], int64_t firstLocalVertex[], int64_t *parentsLocal[]);

/** Restore arrays for the parents of vertices

  \param[in]     numDevices        the number of cuda devices available
  \param[in]     numVerticesLocal  the number of vertices you are assigning to each device
  \param[in]     firstVertexLocal  the first vertex you are assigning to each cuda device
                                   (so device i will have an array for the parents of the vertices
                                   [firstLocalVertex[i], ..., firstLocalVertex[i] + numVerticesLocal[i] -1]).
  \param[out]    parentsLocal      You should deallocate the device memory that was allocated for each device with non-zero number of vertices.
 */
int BFSGraphRestoreParentArraysDevice(BFSGraph graph, int numDevices, size_t numVerticesLocal[], int64_t firstLocalVertex[], int64_t *parentsLocal[]);

/** Create the parent list associated with a lists of keys.

  \param[in]     num_keys          the number of search keys
  \param[in]     key               a list of the keys (duplicated on every MPI process)
  \param[in]     numDevices        the number of cuda devices available
  \param[in]     numVerticesLocal  the number of vertices in the local portion of the output parents array for each device
  \param[in]     firstVertexLocal  the index of the first vertex assigned to each device
  \param[in/out] parentsLocal      parentsLocal[i][d] is the array for key[i] on device d: it has already
                                   been allocated on the device and must be filled in the following manner:
                                     - a key is its own parent: if firstVertexLocal[d] <= key[i] < firstVertexLocal[d] + numVerticesLocal[d], then
                                       parentsLocal[i][d][key[i] - firstVertexLocal[d]] = key[i]
                                     - a vertex j in [firstVertexLocal[d], firstVertexLocal[d] + numVerticesLocal[d]) that cannot be reached by breadth-first search from key[i] has parent -1,
                                       parents[i][d][j - firstVertexLocal[d]] = -1
                                     - otherwise, the parent should be a valid parent in a breadth-first search, so
                                       parents[i][d][j - firstVertexLocal[d]] should be a vertex that is closer to key[i] than j.
 */
int BFSGraphSearchDevice(BFSGraph graph, int num_keys, const int64_t *key, int numDevices, const size_t numVerticesLocal[], const int64_t firstLocalVertex[], int64_t **parentsLocal[]);

#if defined(__cplusplus)
}
#endif
#endif
