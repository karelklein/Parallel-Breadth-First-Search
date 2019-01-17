

CFLAGS    = -O0 -g
FFLAGS    =
SOURCEC   = bfs.c
SOURCECXX =
SOURCEF   =
SOURCEH   = bfs.h
DIRS      =
MANSEC    =
LOCDIR    =

-include Makefile.inc

include ${PETSC_DIR}/lib/petsc/conf/variables

test_bfs_OBJS = test_bfs.o
test_bfs_LIBS = -lbfs

LIBBASE    = libbfs
LIBNAME    = ${LIBBASE}.${AR_LIB_SUFFIX}
CLEANFILES = ${LIBNAME} test_bfs

all: ${LIBNAME} test_bfs

include ${PETSC_DIR}/lib/petsc/conf/rules

${LIBNAME}: lib

test_bfs: ${test_bfs_OBJS} ${LIBNAME} chkopts
	-${CLINKER} -o $@ ${test_bfs_OBJS} ${PETSC_LIB} ${LIBNAME}
	-${DSYMUTIL} $@
	${RM} -f $(test_bfs_OBJS)
