ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) multiple_branches.cu -o multiple_branches.elf
clean:
	rm -f multiple_branches.elf
