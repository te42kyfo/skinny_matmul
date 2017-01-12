NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" #  -Xptxas="-v"

CCFLAGS     := 
LDFLAGS     := -L/opt/cuda/lib64 -L../../magma/lib/ -lcublas -lmagma
INCLUDES 	:= -I../../magma/include/

svqb: svqb.cu ../PseudoQuad.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
