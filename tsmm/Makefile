NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:=
NAME 		:= tsmm
M 			:= 1
N			:= 1
VERSIONS 	:= -DFIX1 -DFIX2 -DFIX_FB -DCUBLAS -DFIX_BLEND -DVAR1
# VERSIONS 	:= -DVAR1
MODE 		:= DR
CONSTANTS	:= -DPARM=$M -DPARN=$N -D$(MODE)=1 $(VERSIONS)
PREFIX		:= .



test: $(PREFIX)/test$M-$N-$(MODE)

run_test: $(PREFIX)/test$M-$N-$(MODE)
	$(PREFIX)/test_$M-$N-$(MODE)

$(PREFIX)/test$M-$N-$(MODE): test.cu Makefile *.cuh
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS)


## perf
run_perf: $(PREFIX)/perf$M-$N-$(MODE)
	$(PREFIX)/perf$M-$N-$(MODE)

perf: $(PREFIX)/perf$M-$N-$(MODE)

$(PREFIX)/perf$M-$N-$(MODE): perf.cu *.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS)

