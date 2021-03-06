NVCC := nvcc

CUDA_HOME ?=  $(CUDATOOLKIT_HOME)

NVCCFLAGS   := -std=c++11 -O3 -arch=sm_60 --compiler-options="-O2 -pipe -Wall -fopenmp -fno-exceptions" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" #  -Xptxas="-v"


CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -L../magma/lib/ -lcublas  -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda
INCLUDES 	:= -I../magma/include -I$(CUDA_HOME)/extras/CUPTI/include
M 			:= 1
N			:= 1

TSMM_VERSIONS 		:= -DFIX3 -DFIX_FB -DCUBLAS
TSMTTSM_VERSIONS 	:= # -DFIX_GENV3 -DFIX_GENV3T -DFIX_GENV4 -DFIX_GENV4T -DFIX_GENV7 -DFIX_GENV8 -DCUBLAS

TYPES 		:= DR
MULTYPE		:= TSMTTSM
GIT_BRANCH_NAME := $(shell git rev-parse --abbrev-ref HEAD)
CONSTANTS	:= -DPARM=$M -DPARN=$N -D$(MULTYPE)=1 -D$(TYPES)=1 $(TSMTTSM_VERSIONS) $(TSMM_VERSIONS) -DVERBOSE_ERRORS -DGIT_BRANCH_NAME="\"$(GIT_BRANCH_NAME)\""
PREFIX		:= ./build/



NAME 		:= -$(MULTYPE)-$M-$N-$(TYPES)

runtest: test
	$(PREFIX)/test$(NAME)

test: $(PREFIX)/test$(NAME)

$(PREFIX)/test$(NAME): test.cu Makefile *.cuh tsmttsm/*.cuh tsmm/*.cuh
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS)

runperf: perf
	$(PREFIX)/perf$(NAME)

perf: $(PREFIX)/perf$(NAME)


$(PREFIX)/perf$(NAME): perf.cu sqlite3.o benchdb.hpp *.cuh tsmttsm/*.cuh tsmm/*.cuh Makefile versions.hpp
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< sqlite3.o $(LDFLAGS)

sqlite3.o: sqlite3.c sqlite3.h
	gcc -O3 sqlite3.c -c -o sqlite3.o

numeric_tests: numeric_tests.cu *.cuh tsmttsm/*.cuh tsmm/*.cuh Makefile versions.hpp
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	rm -f ./build/perf-TSMTTSM-*
	rm -f ./build/perf-TSMM-*
	rm -f ./build/test-TSMTTSM-*
	rm -f ./build/test-TSMM-*
