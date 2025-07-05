ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_cond_dd.cu -o simple_cond_dd.elf
clean:
	rm -f simple_cond_dd.elf
