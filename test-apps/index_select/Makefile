ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) index_select.cu -o index_select.elf

clean:
	rm -f index_select.elf
