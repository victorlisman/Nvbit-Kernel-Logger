ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) composite_loop.cu -o composite_loop.elf
clean:
	rm -f composite_loop.elf
