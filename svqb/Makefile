NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" #  -Xptxas="-v"

CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -L../../magma/lib/ -L../../mpack/lib/ -lcublas -lmagma -lmblas_dd -lmblas_qd  -lmlapack_dd -lmlapack_qd  -lqd
INCLUDES 	:= -I../../magma/include/ -I../../mpack/include/ -I../../mpack/include/mpack

svqb: svqb.cu ../PseudoQuad.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
