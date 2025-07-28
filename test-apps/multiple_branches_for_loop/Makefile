ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) multiple_branches_for_loop.cu -o multiple_branches_for_loop.elf
clean:
	rm -f multiple_branches_for_loop.elf
