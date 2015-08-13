NVCC := nvcc

# internal flags
NVCCFLAGS   := -O3 -arch=sm_35  --compiler-options="-O2 -pipe -march=native -Wall"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64
INCLUDE 	:=
NAME 		:= skinny_matmul

# Target rules
all: test perf

runtest: test
	./test

runperf: perf
	./perf

test: test.o
	$(NVCC) -o $@ $+  $(LDFLAGS)

perf: perf.o
	$(NVCC) -o $@ $+  $(LDFLAGS)

test.o:test.cu matmul.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

perf.o:perf.cu matmul.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<


clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
