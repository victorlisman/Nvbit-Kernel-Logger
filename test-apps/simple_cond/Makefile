ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_cond.cu -o simple_cond.elf
clean:
	rm -f simple_cond.elf
