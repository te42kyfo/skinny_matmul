NVCC := nvcc

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_35 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info -Xcudafe "--diag_suppress=code_is_unreachable" -Xptxas="-v"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64 -lcublas
INCLUDES 	:=
M 			:= 1
N			:= 1
#VERSIONS 	:= -DFIX1 -DFIX2 -DFIX_FB -DCUBLAS -DFIX_BLEND -DVAR1 -DFIXIPG
TSMM_VERSIONS 		:= -DCUBLAS
TSMTTSM_VERSIONS 	:= -DCUBLAS -DFIX_GENV3
TYPES 		:= DR
MODE		:= TSMTTSM
CONSTANTS	:= -DPARM=$M -DPARN=$N -D$(MODE)=1 -D$(TYPES)=1 $(TSMTTSM_VERSIONS) $(TSMM_VERSIONS) -DVERBOSE_ERRORS
PREFIX		:= .

NAME 		:= -$(MODE)-$M-$N-$(TYPES)

test: $(PREFIX)/test$(NAME)

$(PREFIX)/test$(NAME): test.cu Makefile *.cuh
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< $(LDFLAGS)


perf: $(PREFIX)/perf$(NAME)

$(PREFIX)/perf$(NAME): perf.cu ../sqlite3.o ../benchdb.hpp *.cuh Makefile
	$(NVCC) $(NVCCFLAGS) $(CONSTANTS) $(INCLUDES) -o $@ $< sqlite3.o $(LDFLAGS)

sqlite3.o: sqlite3.c sqlite3.h
	gcc -O3 sqlite3.c -c -o sqlite3.o
