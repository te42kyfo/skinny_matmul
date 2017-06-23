NVCC := nvcc

#internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -Wall -fopenmp -g " -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" #  -Xptxas="-v"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -L../magma/lib/ -lcublas  -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda
INCLUDES 	:= -I../magma/include -I$(CUDA_HOME)/extras/CUPTI/include
M 			:= 1
N			:= 1
TSMM_VERSIONS 		:=  -DFIX_FB -DFIX3 -DCUBLAS
TSMTTSM_VERSIONS 	:= -DFIX_GENV8

# -DFIX_GENV1 -DFIX_GENV1T -DFIX_GENV3 -DFIX_GENV3T -DFIX_GENV4 -DFIX_GENV4T -DFIX_SPECSMALL -DFIX_GENV7 -DFIX_GENV8 -DCUBLAS
TYPES 		:= DR
MULTYPE		:= TSMM
GIT_BRANCH_NAME := $(shell git status --porcelain --branch | grep '\#\#' | cut -c 4-)
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
