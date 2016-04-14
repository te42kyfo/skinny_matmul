NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -fopenmp -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:= -I/home/hpc/ihpc/ihpc05/cub/
NAME 		:= skinny_matmul
M 			:= 1
N			:= 1
GENVER 		:= GENV3
PREFIX		:= .


runtest: $(PREFIX)/test$M-$N-$(GENVER)
	$(PREFIX)/test$M-$N-$(GENVER)

runperf: perf
	$(PREFIX)/$<$M-$N-$(GENVER)

perf: perf.cu genv?.cuh skyblas.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N -DSKYBLAS_GENVER=$(GENVER) $(INCLUDES) -o $(PREFIX)/$@$M-$N-$(GENVER)  $<  $(LDFLAGS)


test: $(PREFIX)/test$M-$N-$(GENVER)

$(PREFIX)/test$M-$N-$(GENVER): test.cu genv?.cuh spec8x8.cuh specsym.cuh gen_cublas.cuh skyblas.cuh Makefile
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N -DSKYBLAS_GENVER=$(GENVER) $(INCLUDES) -o $@ $< $(LDFLAGS) --compiler-options="-fopenmp  -g"


clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
