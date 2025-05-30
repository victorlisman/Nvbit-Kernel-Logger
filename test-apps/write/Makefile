ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) write.cu -o write.elf
clean:
	rm -f write.elf
