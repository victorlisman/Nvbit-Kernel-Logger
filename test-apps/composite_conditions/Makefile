ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) composite_conditions.cu -o composite_conditions.elf
clean:
	rm -f composite_conditions.elf
