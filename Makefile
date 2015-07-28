NVCC := nvcc

# internal flags
NVCCFLAGS   := -O3 -arch=sm_35  --compiler-options="-O2 -pipe -march=native -Wall"
CCFLAGS     :=
LDFLAGS     := -L/opt/cuda/lib64
INCLUDE 	:=
NAME 		:= skinny_matmul

# Target rules
all: build

build: $(NAME)

main.o:main.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ -c $<

$(NAME): main.o
	$(NVCC) -o $@ $+  $(LDFLAGS)

run: build
	./$(NAME)

clean:
	rm -f ./$(NAME)
	rm -f main.o
