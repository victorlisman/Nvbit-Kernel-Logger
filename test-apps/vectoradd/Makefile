ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) vectoradd.cu -o vectoradd

clean:
	rm -f vectoradd
