ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) nested_for_loop.cu -o nested_for_loop.elf
clean:
	rm -f nested_for_loop.elf
