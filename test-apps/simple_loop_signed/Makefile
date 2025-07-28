ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_loop_signed.cu -o simple_loop_signed.elf
clean:
	rm -f simple_loop_signed.elf
