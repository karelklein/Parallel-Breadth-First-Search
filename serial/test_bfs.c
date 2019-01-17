const char help[] = "Test driver for the correctness of Breadth-First Search implementation";

#include <petscmat.h>
#include <petscbt.h>
#include "bfs.h"

/* modified from octave file provided with graph500 */
static PetscErrorCode CreateEdgesStochasticKronecker(PetscRandom rand, PetscInt scale, PetscInt num_edges, int64_t (*edges)[2])
{
  PetscReal      a = 0.57;
  PetscReal      b = 0.19;
  PetscReal      c = 0.19;
  PetscReal      ab, c_norm, a_norm;
  PetscInt       p, bit, n = 1 << scale;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ab = a + b;
  c_norm = c / (1. - ab);
  a_norm = a / (ab);
  ierr = PetscRandomSetInterval(rand, 0., 1.);CHKERRQ(ierr);
  for (p = 0; p < num_edges; p++) {
    edges[p][0] = 0;
    edges[p][1] = 0;
    for (bit = 0; bit < scale; bit++) {
      PetscReal r;
      PetscInt  i_b, j_b;

      ierr = PetscRandomGetValueReal(rand, &r);CHKERRQ(ierr);
      i_b = (r > ab);
      ierr = PetscRandomGetValueReal(rand, &r);CHKERRQ(ierr);
      j_b = (r > (c_norm * i_b + a_norm * !(i_b)));
      edges[p][0] |= i_b << bit;
      edges[p][1] |= j_b << bit;
    }
  }

  {
    PetscInt *perm, v;

    ierr = PetscMalloc1(n, &perm);CHKERRQ(ierr);
    for (v = 0; v < n; v++) {perm[v] = v;}
    for (v = 0; v < n; v++) {
      PetscReal rv;
      PetscInt  j, swap;

      ierr = PetscRandomGetValueReal(rand, &rv);CHKERRQ(ierr);
      j = ((PetscInt) (rv * (n - v))) + v;
      swap = perm[v];
      perm[v] = perm[j];
      perm[j] = swap;
    }
    for (v = 0; v < num_edges; v++) {
      edges[v][0] = perm[edges[v][0]];
      edges[v][1] = perm[edges[v][1]];
    }
    ierr = PetscFree(perm);CHKERRQ(ierr);
    for (v = 0; v < num_edges; v++) {
      PetscReal rv;
      PetscInt  j;
      int64_t   swap[2];

      ierr = PetscRandomGetValueReal(rand, &rv);CHKERRQ(ierr);
      j = ((PetscInt) (rv * (num_edges - v))) + v;

      swap[0] = edges[v][0];
      swap[1] = edges[v][1];
      edges[v][0] = edges[j][0];
      edges[v][1] = edges[j][1];
      edges[j][0] = swap[0];
      edges[j][1] = swap[1];
    }
  }

  PetscFunctionReturn(0);
}

static int EdgeCompare(const void *a, const void *b)
{
  const PetscInt *e1 = (const PetscInt *) a;
  const PetscInt *e2 = (const PetscInt *) b;

  if (e1[0] < e2[0]) return -1;
  if (e1[0] > e2[0]) return 1;
  return (e1[1] < e2[1]) ? -1 : ((e1[1] == e2[1]) ? 0 : 1);
}

static PetscErrorCode BreadthFirstSearchCheck(PetscInt n, PetscInt e, const int64_t (*edges)[2], PetscInt key, const int64_t *parents)
{
  PetscInt       edge, v;
  Mat            G;
  PetscInt       *dist, *order;
  PetscInt       vStart, vEnd, rank;
  PetscInt       *nnz;
  PetscInt       (*dedges)[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(2*e,&dedges,n,&nnz);CHKERRQ(ierr);
  for (edge = 0; edge < e; edge++) {
    dedges[2 * edge][0] = edges[edge][0];
    dedges[2 * edge][1] = edges[edge][1];
    dedges[2 * edge + 1][1] = edges[edge][0];
    dedges[2 * edge + 1][0] = edges[edge][1];
  }
  qsort(dedges,2*e,2*sizeof(PetscInt),EdgeCompare);
  for (v = 0; v < n; v++) {nnz[v] = 0;}
  for (edge = 0; edge < 2 * e; edge++) {
    if ((!edge) || (dedges[edge][0] != dedges[edge-1][0]) || (dedges[edge][1] != dedges[edge-1][1])) {
      nnz[dedges[edge][0]]++;
    }
  }
  while (n > 0 && nnz[n - 1] == 0) {n--;}
  ierr = MatCreate(PETSC_COMM_SELF, &G);CHKERRQ(ierr);
  ierr = MatSetType(G, MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(G, n, n, n, n);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(G, 0, nnz);CHKERRQ(ierr);
  for (edge = 0; edge < 2 * e; edge++) {
    PetscInt i = dedges[edge][0];
    PetscInt j = dedges[edge][1];
    ierr = MatSetValue(G, i, j, 1., INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(dedges, nnz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscMalloc2(n, &dist, n, &order);CHKERRQ(ierr);
  for (v = 0; v < n; v++) {dist[v] = -1;}
  rank = 0;
  dist[key] = rank++;
  vStart = vEnd = 0;
  order[vEnd++] = key;
  while (vEnd > vStart) {
    PetscInt vEndOrig = vEnd, v;

    for (v = vStart; v < vEndOrig; v++) {
      PetscInt nneigh, j;
      const PetscInt *neigh;
      PetscInt vert = order[v];

      ierr = MatGetRow(G, vert, &nneigh, &neigh, NULL);CHKERRQ(ierr);
      for (j = 0; j < nneigh; j++) {
        PetscInt neighj = neigh[j];
        if (neighj == vert) continue;
        if (dist[neighj] == -1) {
          dist[neighj] = rank;
          order[vEnd++] = neighj;
        }
      }
      ierr = MatRestoreRow(G, vert, &nneigh, &neigh, NULL);CHKERRQ(ierr);
    }
    rank++;
    vStart = vEndOrig;
  }
  for (v = 0; v < n; v++) {
    int64_t parent = parents[v];

    if (v == key) {
      if (parent != key) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "key %D is not its own parent\n", (PetscInt) key);
      continue;
    }
    if (parent < 0) {
      if (dist[v] >= 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "vertex %D has no parent but has distance %D from key\n", (PetscInt) v, dist[v]);
    } else {
      if (dist[parent] != dist[v] - 1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_LIB, "vertex %D has parent %D but is not one rank farther from the key\n", (PetscInt) v, (PetscInt) parent);
    }
  }
  ierr = PetscFree2(dist, order);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       test, numTests = 10;
  PetscInt       scale = 6;
  PetscInt       factor = 16;
  PetscRandom    rand;
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Breadth-First Search Test Options", "test_bfs.c");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_tests", "Number of tests to run", "test_bfs.c", numTests, &numTests, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-scale", "Scale (log2) of the array in the test", "test_bfs.c", scale, &scale, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-factor", "Ratio of edges to vertices", "test_bfs.c", factor, &factor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer, "Running %D tests of breadth_first_search()\n", numTests);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
  for (test = 0; test < numTests; test++) {
    cse6230graph   graph = NULL;
    PetscInt       n, e, i;
    int64_t        (*edges)[2];
    int64_t        key, *parents;
    PetscBT        connected;

    ierr = PetscViewerASCIIPrintf(viewer, "Test %D:\n", test);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    n = 1 << scale;
    e = factor * n;

    ierr = PetscViewerASCIIPrintf(viewer, "Test: %D edges\n", e);CHKERRQ(ierr);

    ierr = PetscMalloc2(e, &edges, n, &parents);CHKERRQ(ierr);

    ierr = CreateEdgesStochasticKronecker(rand, scale, e, edges);CHKERRQ(ierr);

    ierr = graph_create((size_t) e, edges, &graph);CHKERRQ(ierr);

    ierr = PetscBTCreate(n, &connected);CHKERRQ(ierr);
    for (i = 0; i < e; i++) {
      ierr = PetscBTSet(connected, edges[i][0]);CHKERRQ(ierr);
      ierr = PetscBTSet(connected, edges[i][1]);CHKERRQ(ierr);
    }
    ierr = PetscRandomSetInterval(rand, 0., n);CHKERRQ(ierr);
    while (1) {
      PetscReal rv;

      ierr = PetscRandomGetValueReal(rand, &rv);CHKERRQ(ierr);
      key  = (PetscInt) rv;
      if (PetscBTLookup(connected, key)) break;
    }
    ierr = PetscBTDestroy(&connected);CHKERRQ(ierr);

    ierr = breadth_first_search(graph, 1, &key, &parents);CHKERRQ(ierr);
    ierr = graph_destroy(&graph);CHKERRQ(ierr);

    ierr = BreadthFirstSearchCheck(n, e, edges, key, parents);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
    ierr = PetscViewerASCIIPrintf(viewer, "Passed.\n");CHKERRQ(ierr);

    ierr = PetscFree2(edges, parents);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
    ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
  }
  ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
