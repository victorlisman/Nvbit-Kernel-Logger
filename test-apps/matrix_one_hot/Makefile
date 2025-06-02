ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) matrix_one_hot.cu -o matrix_one_hot.elf

clean:
	rm -f matrix_one_hot.elf
