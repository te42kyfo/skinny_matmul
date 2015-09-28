NVCC := nvcc

# internal flags
NVCCFLAGS   := -O3 -arch=sm_35  --compiler-options="-O2 -pipe -march=native -Wall -fopenmp"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:= -I/home/hpc/ihpc/ihpc05/cub/
NAME 		:= skinny_matmul
M 			:= 1
N			:= 1
GENVER 		:= GENV3
PREFIX		:= .


runtest: test
	./test

test: test.o
	$(NVCC) -o $@ $+  $(LDFLAGS)  --compiler-options="-fopenmp"

perf: perf.cu genv?.cuh
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N -DGENVER=$(GENVER) $(INCLUDES) -o $(PREFIX)/$@$M-$N-$(GENVER)  $<  $(LDFLAGS)

test.o:test.cu genv1.cuh genv2.cuh genv3.cuh genv4.cuh multi_dispatch.cuh
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ -c $<


clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
