ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) int_one_hot.cu -o int_one_hot.elf
clean:
	rm -f int_one_hot.elf
