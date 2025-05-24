ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) readWrite.cu -o readWrite

clean:
	rm -f readWrite
