NVCC := nvcc

# internal flags
NVCCFLAGS   := -O3 -arch=sm_35  --compiler-options="-O2 -pipe -march=native -Wall"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64
INCLUDE 	:=
NAME 		:= skinny_matmul
M 			:= 1
N			:= 1
PREFIX		:= .

# Target rules
all: test perf

runtest: test
	./test

runperf: perf
	$(PREFIX)/perf$M-$N

test: test.o
	$(NVCC) -o $@ $+  $(LDFLAGS)

perf: $(PREFIX)/perf$M-$N.o
	$(NVCC) -o $(PREFIX)/$@$M-$N $+ $(LDFLAGS)

test.o:test.cu matmul.cuh
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ -c $<

$(PREFIX)/perf$M-$N.o:perf.cu matmul.cuh
	$(NVCC) $(NVCCFLAGS) -DPARM=$M -DPARN=$N $(INCLUDES) -o $@ -c $<



clean:
	rm -f ./$(NAME)
	rm -f main.o
	rm -f test.o
	rm -f perf.o
	rm -f perf
	rm -f test
