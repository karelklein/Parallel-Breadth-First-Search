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

static PetscErrorCode BreadthFirstSearchCheck(PetscInt n, PetscInt e, const int64_t (*edges)[2], PetscInt key, const int64_t *parents, PetscBool *passed)
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
      if (parent != key) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "key %D is not its own parent\n", (PetscInt) key); CHKERRQ(ierr);
        *passed = PETSC_FALSE;
        break;
      }
      else {
        continue;
      }
    }
    if (parent < 0) {
      if (dist[v] >= 0) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "vertex %D has no parent but has distance %D from key\n", (PetscInt) v, dist[v]); CHKERRQ(ierr);
        *passed = PETSC_FALSE;
        break;
      }
    } else {
      if (dist[parent] != dist[v] - 1) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "vertex %D has parent %D but is not one rank farther from the key\n", (PetscInt) v, (PetscInt) parent); CHKERRQ(ierr);
        *passed = PETSC_FALSE;
        break;
      }
    }
  }
  ierr = PetscFree2(dist, order);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscInt       tests[4] = {14, 17, 20, 23}, test;
  double         pps[4];
  PetscInt       numTests = 4;
  PetscInt       nCheck = 10;
  PetscInt       factor = 16;
  PetscRandom    rand;
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "Breadth-First Search Test Options", "test_bfs.c");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-tests", "Test sizes to run", "test_bfs.c", tests, &numTests, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_time", "The number of times a test is timed", "test_bfs.c", nCheck, &nCheck, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-factor", "Ratio of edges to vertices", "test_bfs.c", factor, &factor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!numTests) {
    numTests = 4;
  }

  ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = PetscRandomSeed(rand);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer, "Running %D tests of breadth first search\n", numTests);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);
  for (test = 0; test < numTests; test++) {
    BFSGraph       graph = NULL;
    PetscInt       n, e;
    int            eLocalInt, vLocalInt, *eLocalV = NULL, *dispv = NULL, *vLocalV = NULL;
    int64_t        (*edges)[2], (*edgesGlobal)[2] = NULL;
    int64_t        key, **parents, *parentsGlobal = NULL;
    int64_t        *keys;
    size_t         nvGlobal;
    int            size, rank;
    size_t         nvLocal, eLocal;
    int64_t        firstLocal;
    int64_t        numConnected = 0;
    int            numKeys;
    double         totalTime = 0.;
    PetscBool      passed = PETSC_TRUE;
    PetscBT        connected;
    PetscInt       i;

    ierr = PetscViewerASCIIPrintf(viewer, "Test %D: scale %D\n", test, tests[test]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = BFSGraphCreate(comm, &graph); CHKERRQ(ierr);

    n = 1 << tests[test];
    e = factor * n;

    ierr = PetscViewerASCIIPrintf(viewer, "Test: %D edges\n", e);CHKERRQ(ierr);

    ierr = BFSGraphGetEdgeArray(graph, e, &eLocal, &edges); CHKERRQ(ierr);

    ierr = CreateEdgesStochasticKronecker(rand, tests[test], eLocal, edges);CHKERRQ(ierr);

    ierr = BFSGraphSetEdges(graph, (size_t) e, eLocal, (const int64_t (*)[2]) edges);CHKERRQ(ierr);

    ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    if (!rank) {
      ierr = PetscMalloc1(e, &edgesGlobal); CHKERRQ(ierr);
      ierr = PetscMalloc2(size, &eLocalV, size, &dispv); CHKERRQ(ierr);
    }
    eLocalInt = eLocal;
    ierr = MPI_Gather(&eLocalInt, 1, MPI_INT, eLocalV, 1, MPI_INT, 0, comm); CHKERRQ(ierr);
    if (!rank) {
      dispv[0] = 0;
      for (int p = 0; p < size; p++) {
        eLocalV[p] *= 2;
        if (p < size - 1) {
          dispv[p + 1] = dispv[p] + eLocalV[p];
        }
      }
    }
    ierr = MPI_Gatherv(edges, 2 * eLocal, MPI_INT64_T, edgesGlobal, eLocalV, dispv, MPI_INT64_T, 0, comm);CHKERRQ(ierr);
    ierr = PetscMalloc2(64,&keys,64,&parents);CHKERRQ(ierr);
    if (!rank) {
      ierr = PetscBTCreate(n, &connected);CHKERRQ(ierr);
      for (int i = 0; i < e; i++) {
        int64_t v = edgesGlobal[i][0];

        if (edgesGlobal[i][0] == edgesGlobal[i][1]) continue;

        if (!PetscBTLookup(connected, v)) {
          ierr = PetscBTSet(connected, v);CHKERRQ(ierr);
          numConnected++;
        }

        v = edgesGlobal[i][1];

        if (!PetscBTLookup(connected, v)) {
          ierr = PetscBTSet(connected, v);CHKERRQ(ierr);
          numConnected++;
        }
      }
      ierr = PetscRandomSetInterval(rand, 0., n);CHKERRQ(ierr);
      numKeys = PetscMin(64, numConnected);
      for (int k = 0; k < numKeys;) {
        int kj;

        while (1) {
          PetscReal rv;

          ierr = PetscRandomGetValueReal(rand, &rv);CHKERRQ(ierr);
          key  = (PetscInt) rv;
          if (PetscBTLookup(connected, key)) break;
        }
        for (kj = 0; kj < k; kj++) {
          if (keys[kj] == key) break;
        }
        if (kj == k) {
          keys[k++] = key;
        }
      }
      ierr = PetscBTDestroy(&connected);CHKERRQ(ierr);
      ierr = PetscFree2(eLocalV,dispv);CHKERRQ(ierr);
      for (int k = numKeys; k < 64; k++) {
        keys[k] = -1;
      }
    }
    ierr = MPI_Bcast(keys, 64, MPI_INT64_T, 0, comm);CHKERRQ(ierr);

    for (numKeys = 0; numKeys < 64; numKeys++) {
      if (keys[numKeys] == -1) break;
    }

    for (int k = 0; k < numKeys; k++) {
      ierr = BFSGraphGetParentArray(graph, &nvLocal, &firstLocal, &parents[k]);CHKERRQ(ierr);
    }
    for (i = 0; i < nCheck + 1; i++) {
      double tic, toc, total, maxTotal = 0.;

      for (int k = 0; k < numKeys; k++) {
        memset(parents[k], -1, nvLocal * sizeof (int64_t)); CHKERRQ(ierr);
      }
      ierr = MPI_Barrier(comm); CHKERRQ(ierr);
      tic = MPI_Wtime();
      ierr = BFSGraphSearch(graph, numKeys, keys, nvLocal, firstLocal, parents);CHKERRQ(ierr);
      toc = MPI_Wtime();
      total = toc - tic;
      ierr = MPI_Allreduce(&total, &maxTotal, 1, MPI_DOUBLE, MPI_MAX, comm); CHKERRQ(ierr);
      if (i) {totalTime += maxTotal;}
    }
    totalTime /= nCheck;

    if (!rank) {
      ierr = PetscMalloc2(size, &vLocalV, size, &dispv); CHKERRQ(ierr); CHKERRQ(ierr);
    }
    vLocalInt = nvLocal;
    ierr = MPI_Gather(&vLocalInt, 1, MPI_INT, vLocalV, 1, MPI_INT, 0, comm); CHKERRQ(ierr);
    nvGlobal = 0;
    if (!rank) {
      dispv[0] = 0;
      for (int p = 0; p < size; p++) {
        nvGlobal += vLocalV[p];
        if (p < size - 1) {
          dispv[p + 1] = dispv[p] + vLocalV[p];
        }
      }
      ierr = PetscMalloc1(nvGlobal, &parentsGlobal);CHKERRQ(ierr);
    }
    for (int k = 0; k < numKeys; k++) {
      ierr = MPI_Gatherv(parents[k], nvLocal, MPI_INT64_T, parentsGlobal, vLocalV, dispv, MPI_INT64_T, 0, comm);CHKERRQ(ierr);
      if (!rank) {
        if (nvGlobal < numConnected) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "Misidentified number of connected vertices (%D < %D)\n", nvGlobal, numConnected); CHKERRQ(ierr);
          passed = PETSC_FALSE;
        } else {
          ierr = BreadthFirstSearchCheck(nvGlobal, e, (const int64_t (*)[2]) edgesGlobal, keys[k], parentsGlobal, &passed);CHKERRQ(ierr);
        }
      }
    }
    if (!rank) {
      ierr = PetscFree(parentsGlobal);CHKERRQ(ierr);
      ierr = PetscFree2(vLocalV, dispv);CHKERRQ(ierr);
    }

    ierr = MPI_Bcast(&passed, 1, MPIU_BOOL, 0, comm); CHKERRQ(ierr);

    if (passed) {
      ierr = PetscViewerASCIIPrintf(viewer, "Test scale %D, %D keys: Passed, average time (%D tests): %g, ***%g parents per second***.\n", tests[test], numKeys, nCheck, totalTime, (n * numKeys) / totalTime);CHKERRQ(ierr);
      pps[test] = (n * numKeys) / totalTime;
    } else {
      ierr = PetscViewerASCIIPrintf(viewer, "Test scale %D, %D keys: FAILED, average time (%D tests): %g, ***0 parents per second***.\n", tests[test], numKeys, nCheck, totalTime, (n * numKeys) / totalTime);CHKERRQ(ierr);
      pps[test] = 0;
    }

    for (int k = 0; k < numKeys; k++) {
      ierr = BFSGraphRestoreParentArray(graph, &nvLocal, &firstLocal, &parents[k]);CHKERRQ(ierr);
    }
    ierr = BFSGraphRestoreEdgeArray(graph, e, &eLocal, &edges); CHKERRQ(ierr);
    ierr = PetscFree2(keys, parents);CHKERRQ(ierr);

    if (!rank) {
      ierr = PetscFree(edgesGlobal);CHKERRQ(ierr);
    }

    ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
    ierr = BFSGraphDestroy(&graph); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  {
    double harmonicMean = 0.;

    for (test = 0; test < numTests; test++) {
      harmonicMean += 1./(pps[test] + PETSC_SMALL);
    }
    harmonicMean = numTests / harmonicMean;

    ierr = PetscViewerASCIIPrintf(viewer, "===Harmonic mean: %g parents per second===\n", harmonicMean); CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}
