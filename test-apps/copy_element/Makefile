ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) copy_element.cu -o copy_element.elf

clean:
	rm -f copy_element.elf
