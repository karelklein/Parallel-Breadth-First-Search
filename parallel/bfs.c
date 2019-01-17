#include <petscsys.h>
#include <mpi.h>
#include "bfs.h"

struct _bfsgraph
{
  MPI_Comm comm;
  int64_t n_vertices_local;
  int64_t n_vertices_global;
  int64_t n_edges_local;
  int64_t (*my_edges)[2];
};

int BFSGraphCreate(MPI_Comm comm, BFSGraph *graph_p)
{
  BFSGraph       graph = NULL;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscCalloc1(1,&graph); CHKERRQ(ierr);

  graph->comm = comm;
  graph->n_vertices_local = 0;
  graph->n_vertices_global = 0;
  graph->n_edges_local = 0;

  *graph_p = graph;
  PetscFunctionReturn(0);
}

int BFSGraphDestroy(BFSGraph *graph_p)
{
  PetscFunctionBeginUser;

  free((*graph_p)->my_edges);
  PetscFree(*graph_p);
  *graph_p = NULL;

  PetscFunctionReturn(0);
}

int BFSGraphGetEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**elocal_p)[2])
{
  MPI_Comm       comm;
  int            size, rank;
  size_t         edgeStart, edgeEnd;
  size_t         numLocal;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  comm = graph->comm;

  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

  edgeStart = (rank * E) / size;
  edgeEnd   = ((rank + 1) * E) / size;
  numLocal  = edgeEnd - edgeStart;

  ierr = PetscMalloc1(numLocal, elocal_p); CHKERRQ(ierr);
  *numEdgesLocal = numLocal;

  // initialize elocal_p
  for (size_t i = 0; i < numLocal; i++) {
      (*elocal_p)[i][0] = -1;
      (*elocal_p)[i][1] = -1;
  }

  PetscFunctionReturn(0);
}

int BFSGraphRestoreEdgeArray(BFSGraph graph, size_t E, size_t *numEdgesLocal, int64_t (**elocal_p)[2])
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  MPI_Comm  comm;
  int       size;
  int       rank;
  size_t    edgeStart, edgeEnd;
  size_t    numLocal;

  comm = graph->comm;

  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

  edgeStart = (rank * E) / size;
  edgeEnd   = ((rank + 1) * E) / size;
  numLocal  = edgeEnd - edgeStart;

  // free elocal_p arrays
  ierr = PetscFree(*elocal_p); CHKERRQ(ierr);

  *elocal_p = NULL;
  *numEdgesLocal = 0;

  PetscFunctionReturn(0);
}

int BFSGraphSetEdges(BFSGraph graph, size_t E, size_t numEdgesLocal, const int64_t (*eLocal_p)[2])
{
    PetscFunctionBeginUser;

    MPI_Comm  comm;
    int       size;
    int       rank;

    comm = graph->comm;

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // make a copy of the edges
    const int64_t (*eLocal_copy)[2];
    eLocal_copy = eLocal_p;

    // find edge with highest ID
    int64_t biggest = 0;
    for (int e = 0; e < numEdgesLocal; e++) {
        if (eLocal_copy[e][0] > biggest) {biggest = eLocal_copy[e][0];}
        if (eLocal_copy[e][1] > biggest) {biggest = eLocal_copy[e][1];}
        }

    // set the number of vertices in this process' graph
    int64_t N = biggest + 1;
    // graph->n_vertices_local = N;

    // Allreduce the number of local vertices in order to find the number of global vertices
    int64_t n_vertices_global;
    MPI_Allreduce(&N, &n_vertices_global, 1, MPI_INT64_T, MPI_MAX, comm);
    graph->n_vertices_global = n_vertices_global;

    graph->n_vertices_local = n_vertices_global / size;

    /* distribute edges of graph so each processor owns roughly n/p vertices
    and all the outgoing edges from those vertices */

    // each process will have p arrays of size numEdgesLocal to organize the edges it needs to send.
    // edges_to_send will take on a dimension of p x numEdgesLocal x 2
    // NOTE there may well be a better way to do this

    int64_t ***edges_to_send;
    edges_to_send = (int64_t***) malloc(size * sizeof(int64_t**));

    for (int p = 0; p < size; p++) {
        edges_to_send[p] = (int64_t**) malloc(2 * numEdgesLocal * sizeof(int64_t*));
    }

    for (int p = 0; p < size; p++) {
        for (int e = 0; e < 2 * numEdgesLocal; e++) {
            edges_to_send[p][e] = (int64_t*) malloc(2 * sizeof(int64_t));
        }
    }

    // initialize edges_to_send
    for (int p = 0; p < size; p++) {
        for (int64_t i = 0; i < 2 * numEdgesLocal; i++) {
            edges_to_send[p][i][0] = -1;
            edges_to_send[p][i][1] = -1;
        }
    }

    // get the number of vertices assigned to each process
    int64_t num_vert_per_proc;
    num_vert_per_proc = n_vertices_global / size;

    // keep counts of the number of edges to send to each process
    int *counts;
    counts = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        counts[i] = 0;
    }

    // iterate through local edges and place each in appropriate bin
    int64_t u, v;
    for (int64_t i = 0; i < numEdgesLocal; i++) {
        u = eLocal_copy[i][0];
        v = eLocal_copy[i][1];

        // (u,v)
        int owner;
        int count;
        owner = u / num_vert_per_proc;
        count = counts[owner];

        edges_to_send[owner][count][0] = u;
        edges_to_send[owner][count][1] = v;

        counts[owner]++;

        // (v,u)
        owner = v / num_vert_per_proc;
        count = counts[owner];
        edges_to_send[owner][count][0] = v;
        edges_to_send[owner][count][1] = u;

        counts[owner]++;
    }
    // double counts
    for (int i = 0; i < size; i++) {
        counts[i] *= 2;
    }

    // make an array of send displacements corresponding to the counts array
    int *send_disp;
    send_disp = (int*) malloc(size * sizeof(int));
    send_disp[0] = 0;
    for (int i = 1; i < size; i++) {
        send_disp[i] = send_disp[i-1] + counts[i-1];
    }

    // not the most efficient approach, but will now throw all edges into one send array
    // such that processes' sets of edges will be contiguous in the array:
    // [  p1's edges , p2's edges, ... , pn's edges]
    int64_t c = 0;
    int64_t (*send_edges)[2];
    send_edges = malloc(2 * 2 * numEdgesLocal * sizeof(int64_t));

    for (int p = 0; p < size; p++) {
        for (int64_t e = 0; e < 2 * numEdgesLocal; e++) {
            // if edge exists, add it to send_edges
            if (edges_to_send[p][e][0] != -1) {
                send_edges[c][0] = edges_to_send[p][e][0];
                send_edges[c][1] = edges_to_send[p][e][1];
                c++;
            }
        }
    }

    // free edges_to_send. no longer needed.
    for (int p = 0; p < size; p++) {
        for (int64_t i = 0; i < 2 * numEdgesLocal; i++) {
            free(edges_to_send[p][i]);
        }
        free(edges_to_send[p]);
    }
    free(edges_to_send);

    // Alltoall the counts into recv_counts
    int *recv_counts;
    recv_counts = (int*) malloc(size * sizeof(int));
    MPI_Alltoall(counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

    // make an array of recv displacements corresponding to rec_counts
    int *recv_disp;
    recv_disp = (int*) malloc(size * sizeof(int));
    recv_disp[0] = 0;
    for (int i = 1; i < size; i++) {
        recv_disp[i] = recv_disp[i-1] + recv_counts[i-1];
    }

    // get the total number of edges to be received for this process
    int64_t total_edges_recvd = 0;
    for (int i = 0; i < size; i++) {
        total_edges_recvd += recv_counts[i];
    }
    total_edges_recvd /= 2;

    // allocate an array of size total_edges_recvd to receive edges
    int64_t (*recv_edges)[2];
    recv_edges = malloc(2 * total_edges_recvd * sizeof(int64_t));

    // AlltoAllv so each process receives the edges it will own
    MPI_Alltoallv(send_edges, counts, send_disp, MPI_INT64_T,
                  recv_edges, recv_counts, recv_disp,
                  MPI_INT64_T, comm);

    /***
    All processes should now own approx n/p edges.
    For a given edge (u,v), the owner process can be found by: u / num_vert_per_proc
    ***/

    /* Now, utilize the edges to make a local graph structure.
    for now, just keep the new local edges as the structure */

    graph->my_edges = recv_edges;
    graph->n_edges_local = total_edges_recvd;

    free(counts);
    free(send_disp);
    free(send_edges);
    free(recv_counts);
    free(recv_disp);

    PetscFunctionReturn(0);

}

int BFSGraphGetParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal)
{
  // NOTE: I think this has to be modified to match the number of vertices owned by this process (get this # from graph)
  PetscFunctionBeginUser;
  PetscErrorCode ierr;

  MPI_Comm  comm;
  int       size;
  int       rank;
  size_t    vertexStart;
  int64_t    n_vertices_global;
  int64_t    n_vertices_local;

  comm = graph->comm;
  n_vertices_local = graph->n_vertices_local;
  n_vertices_global = graph->n_vertices_global;

  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

  // allocate memory for local parents array
  ierr = PetscMalloc1(n_vertices_local, parentsLocal); CHKERRQ(ierr);

  // get the number of vertices for local parents array
  vertexStart = (n_vertices_global * rank) / size;

  *numVerticesLocal = n_vertices_local;
  *firstLocalVertex = vertexStart;

  PetscFunctionReturn(0);
}

int BFSGraphRestoreParentArray(BFSGraph graph, size_t *numVerticesLocal, int64_t *firstLocalVertex, int64_t **parentsLocal)
{
  PetscFunctionBeginUser;
  PetscErrorCode ierr;

  MPI_Comm  comm;
  int       size;
  int       rank;
  size_t    vertexStart, vertexEnd;
  size_t    numVertices_local;
  size_t    n_vertices;

  comm = graph->comm;
  n_vertices = graph->n_vertices_global;

  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

  // get the number of vertices for local parents array
  vertexStart = (n_vertices * rank) / size;
  vertexEnd  = (n_vertices * (rank + 1) / size);
  numVertices_local = vertexEnd - vertexStart;

  // deallocate array of parents
  ierr = PetscFree(*parentsLocal); CHKERRQ(ierr);
  *numVerticesLocal = numVertices_local;
  *firstLocalVertex = vertexStart;

  PetscFunctionReturn(0);
}

int BFSGraphSearch(BFSGraph graph, int num_keys, const int64_t *key, size_t numVerticesLocal, int64_t firstLocalVertex, int64_t **parentsLocal)

{
  PetscFunctionBeginUser;

  MPI_Comm  comm;
  int       size;
  int       rank;
  int64_t   n_vertices_local;
  int64_t   n_vertices_global;

  comm                = graph->comm;
  n_vertices_local    = graph->n_vertices_local;
  n_vertices_global   = graph->n_vertices_global;

  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  int *send_disp;
  int *recv_adj_counts;
  int *recv_disp;
  int *num_adj_per_proc;

  send_disp =        (int*) malloc(size * sizeof(int));
  recv_adj_counts =  (int*) malloc(size * sizeof(int));
  recv_disp =        (int*) malloc(size * sizeof(int));
  num_adj_per_proc = (int*) malloc(size * sizeof(int));

  // make an set of frontier vertices and next vertices.
  int64_t *FS;
  FS = (int64_t*) malloc(n_vertices_local * sizeof(int64_t));

  // iterate through keys
  for (int k = 0; k < num_keys; k++) {

      // the current key is the source
      int64_t source = key[k];

      // initialize FS
      for (int64_t i = 0; i < n_vertices_local; i++) {
          FS[i] = -2;
      }

      // retrieve the edges owned by this process
      int64_t (*my_edges)[2];
      my_edges = graph->my_edges;

      // number of vertices per process
      int64_t num_vert_per_proc;
      num_vert_per_proc = n_vertices_global / size;

      // find process who owns source
      int owner;
      owner = source / num_vert_per_proc;

      // initialize a head and tail for FS
      int64_t head, tail = 0;

      // if this process owns source, push it onto FS. update its parent
      if (owner == rank) {
          FS[tail] = source;
          parentsLocal[k][source - firstLocalVertex] = source;
          tail++;
      }
      // while FS is not empty
      for (head = 0; head < n_vertices_local; head++) {

          // flag signaling process is done
          int isDone = 0;
          if (head >= tail) isDone = 1;

          // terminating condition (if all processes are done, break)
          int globalDone = 0;
          MPI_Allreduce(&isDone, &globalDone, 1, MPI_INT, MPI_LAND, comm);

          if (globalDone) break;

           /* 1. local discovery: explore adjacencies of vertices in current frontier */

          // get number of local edges,
          int64_t num_edges_local;
          num_edges_local = graph->n_edges_local;

          // now, allocate an array to hold the local adjacencies per proc. At most, there are
          // num_edges_local adjacencies from this process. This will hold the 'v's of (u,v)
          // make a similar array that will hold the 'u's of (u,v)

          int64_t **local_adj, **local_adj_parents;
          local_adj         = (int64_t**) malloc(size * sizeof(int64_t*));
          local_adj_parents = (int64_t**) malloc(size * sizeof(int64_t*));

          for (int p = 0; p < size; p++) {
              // initialize array of counters
              num_adj_per_proc[p] = 0;

              local_adj[p]         = (int64_t*) malloc(num_edges_local * sizeof(int64_t));
              local_adj_parents[p] = (int64_t*) malloc(num_edges_local * sizeof(int64_t));

              // initialize
              for (int64_t n = 0; n < num_edges_local; n++) {
                  local_adj[p][n] = -1;
                  local_adj_parents[p][n] = -1;
              }
          }

          // keep the total number of adjacencies this process found
          int64_t total_num_adjacencies = 0;

          // for each vertex u in FS, add the adjacent vertex v to local_adj
          // and the parent u to local_adj_parents
          for (int64_t i = head; i < tail; i++) {
              int64_t u, v;
              u = FS[i];

              for (int64_t e = 0; e < num_edges_local; e++) {
                  if (my_edges[e][0] == u) {
                      v = my_edges[e][1];
                      total_num_adjacencies++;

                      // get the owner of v
                      int owner_v;
                      owner_v = v / num_vert_per_proc;
                      int64_t index;
                      index = num_adj_per_proc[owner_v];

                      local_adj[owner_v][index] = v;
                      local_adj_parents[owner_v][index] = u;
                      num_adj_per_proc[owner_v]++;
                  }
              }
          }
          // need to merge local_adj into one array, where vertices owned by a process are contiguous in the array.
          int64_t *send_adjacencies;
          send_adjacencies = (int64_t*) malloc(total_num_adjacencies * sizeof(int64_t));

          // same for local_adj_parents
          int64_t *send_parents_of_adj;
          send_parents_of_adj = (int64_t*) malloc(total_num_adjacencies * sizeof(int64_t));

          int64_t ix = 0;
          for (int p = 0; p < size; p++) {
              for (int64_t n = 0; n < num_edges_local; n++) {
                  if (local_adj[p][n] != -1) {
                      send_adjacencies[ix] = local_adj[p][n];
                      send_parents_of_adj[ix] = local_adj_parents[p][n];
                      ix++;
                  }
              }
          }

          // free local_adj and local_adj_parents. no longer needed
          for (int p = 0; p < size; p++) {
              free(local_adj[p]);
              free(local_adj_parents[p]);
          }
          free(local_adj);
          free(local_adj_parents);

          /* 2. Alltoallv exchange of adjacencies */

          // first, make an array of send displacements based on num_adj_per_proc
          send_disp[0] = 0;
          for (int i = 1; i < size; i++) {
              send_disp[i] = send_disp[i-1] + num_adj_per_proc[i-1];
          }

          // Alltoall the adjacency counts into recv_adj_counts
          MPI_Alltoall(num_adj_per_proc, 1, MPI_INT, recv_adj_counts, 1, MPI_INT, comm);

          // make an array of recv displacements based on recv_adj_counts
          recv_disp[0] = 0;
          for (int i = 1; i < size; i++) {
              recv_disp[i] = recv_disp[i-1] + recv_adj_counts[i-1];
          }

          // get the number of vertices to be received for this process
          int64_t num_adj_received = 0;
          for (int i = 0; i < size; i++) {
              num_adj_received += recv_adj_counts[i];
          }

          // allocate space for the adjacencies to be received (v of (u,v))
          int64_t *recv_adjacencies;
          recv_adjacencies = (int64_t*) malloc(num_adj_received * sizeof(int64_t));

          // Alltoallv exchange of adjacencies
          MPI_Alltoallv(send_adjacencies, num_adj_per_proc, send_disp, MPI_INT64_T,
                        recv_adjacencies, recv_adj_counts, recv_disp,
                        MPI_INT64_T, comm);

          // allocate space for the parents of adjacencies to be received (u of (u,v))
          int64_t *recv_parents_of_adj;
          recv_parents_of_adj = (int64_t*) malloc(num_adj_received * sizeof(int64_t));

          //Alltoallv exchange of parents of adjacencies
          MPI_Alltoallv(send_parents_of_adj, num_adj_per_proc, send_disp, MPI_INT64_T,
                        recv_parents_of_adj, recv_adj_counts, recv_disp,
                        MPI_INT64_T, comm);

          /* 3. Local update: Update distances/parents for unvisited vertices. */

          // for each vertex v in recv_adjacencies, if v has not been visited, update its parents
          for (int64_t i = 0; i < num_adj_received; i++) {
              int64_t u = recv_parents_of_adj[i];
              int64_t v = recv_adjacencies[i];


              if (parentsLocal[k][v - firstLocalVertex] == -1) {
                  parentsLocal[k][v - firstLocalVertex] = u;

                //   if (u >0) {
                //       printf("rank: %d, size: %d, k: %d, source: %lld\n", rank, size, k, source);
                //       printf("num vert per proc: %lld, firstLocalVertex: %lld\n", num_vert_per_proc, firstLocalVertex);
                //       printf("numVerticesLocal: %zu, graph->n_vertices_local: %lld, n_vertices_global: %lld\n", numVerticesLocal, n_vertices_local, n_vertices_global);
                //       printf("rank: %d, (u, v): %lld, %lld\n", rank, u, v);
                //       printf("after: parentsLocal[k][v-firstLocalVertex] = %lld\n\n", parentsLocal[k][v - firstLocalVertex]);
                //   }

                  // add v to the frontier FS
                  FS[tail] = v;
                  tail++;
              }
          }

          free(send_adjacencies);
          free(send_parents_of_adj);
          free(recv_adjacencies);
          free(recv_parents_of_adj);
      }
  }

  free(send_disp);
  free(recv_disp);
  free(num_adj_per_proc);
  free(recv_adj_counts);
  free(FS);

  PetscFunctionReturn(0);
}
