ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) readWrite.cu -o readWrite.elf

clean:
	rm -f readWrite.elf
