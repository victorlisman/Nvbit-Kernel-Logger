ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) reverse_index_copy.cu -o reverse_index_copy.elf

clean:
	rm -f reverse_index_copy.elf
