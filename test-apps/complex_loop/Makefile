ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) complex_loop.cu -o complex_loop.elf
clean:
	rm -f complex_loop.elf
