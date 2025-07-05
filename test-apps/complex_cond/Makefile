ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) complex_cond.cu -o complex_cond.elf
clean:
	rm -f complex_cond.elf
