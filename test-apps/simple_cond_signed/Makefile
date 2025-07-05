ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) simple_cond_signed.cu -o simple_cond_signed.elf
clean:
	rm -f simple_cond_signed.elf

