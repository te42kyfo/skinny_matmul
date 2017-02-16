NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" 

CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64  -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include

membw: membw.cu Makefile
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
