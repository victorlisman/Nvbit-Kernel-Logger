ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_loop_dd.cu -o simple_loop_dd.elf
clean:
	rm -f simple_loop_dd.elf
