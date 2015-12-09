NVCC := nvcc

# internal flags
NVCCFLAGS   := -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic -lineinfo
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:= -I/home/hpc/ihpc/ihpc05/cub/
NAME 		:= skinny_matmul
M 			:= 1
N			:= 1
GENVER 		:= GENV3
PREFIX		:= .


runtest: test
	$(PREFIX)/$<$M-$N-$(GENVER)

runperf: perf
	$(PREFIX)/$<$M-$N-$(GENVER)

perf: perf.cu genv?.cuh skyblas.cuh
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N -DSKYBLAS_GENVER=$(GENVER) $(INCLUDES) -o $(PREFIX)/$@$M-$N-$(GENVER)  $<  $(LDFLAGS)

test: test.cu genv?.cuh skyblas.cuh
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N -DSKYBLAS_GENVER=$(GENVER) $(INCLUDES) -o $(PREFIX)/$@$M-$N-$(GENVER)  $<  $(LDFLAGS) --compiler-options="-fopenmp  -g"


clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
