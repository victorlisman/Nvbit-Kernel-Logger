ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_loop.cu -o simple_loop.elf
clean:
	rm -f simple_loop.elf
