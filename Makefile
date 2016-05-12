NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:= -I/home/hpc/ihpc/ihpc05/cub/
NAME 		:= skinny_matmul
M 			:= 1
N			:= 1
GENVER 		:= GENV3
MODE 		:= DR
CONSTANTS	:= -DPARM=$M -DPARN=$N -DSKYBLAS_GENVER=$(GENVER) -D$(MODE)=1
PREFIX		:= .

# perf
######

runperf: perf
	$(PREFIX)/$<$M-$N-$(GENVER)

perf: perf.cu genv?.cuh skyblas.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $(PREFIX)/$@$M-$N-$(GENVER)-$(MODE)  $<  $(LDFLAGS)

runtest: $(PREFIX)/test$M-$N-$(GENVER)
	$(PREFIX)/test$M-$N-$(GENVER)


# test
######

test: $(PREFIX)/test$M-$N-$(GENVER)-$(MODE)

$(PREFIX)/test$M-$N-$(GENVER)-$(MODE): test.cu genv?.cuh spec8x8.cuh specsmall.cuh gen_cublas.cuh skyblas.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS) --compiler-options="-fopenmp  -g"


# test_tsmm
###########

run_test_tsmm: $(PREFIX)/test_tsmm$M-$N-$(MODE)
	$(PREFIX)/test_tsmm$M-$N-$(MODE)

test_tsmm: $(PREFIX)/test_tsmm$M-$N-$(MODE)

$(PREFIX)/test_tsmm$M-$N-$(MODE): test_tsmm.cu tsmm.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS) --compiler-options="-fopenmp  -g"


# perf_tsmm
###########

run_perf_tsmm: $(PREFIX)/perf_tsmm$M-$N-$(MODE)
	$(PREFIX)/perf_tsmm$M-$N-$(MODE)

perf_tsmm: $(PREFIX)/perf_tsmm$M-$N-$(MODE)

$(PREFIX)/perf_tsmm$M-$N-$(MODE): perf_tsmm.cu tsmm.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS) --compiler-options="-fopenmp  -g"



clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
