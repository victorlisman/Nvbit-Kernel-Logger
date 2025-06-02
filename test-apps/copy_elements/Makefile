ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) copy_elements.cu -o copy_elements.elf

clean:
	rm -f copy_elements.elf
