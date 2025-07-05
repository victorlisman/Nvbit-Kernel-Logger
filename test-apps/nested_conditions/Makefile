ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) nested_conditions.cu -o nested_conditions.elf
clean:
	rm -f nested_conditions.elf
