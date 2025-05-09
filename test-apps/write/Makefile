ARCH=sm_70

CXX = /usr/bin/g++-13
all:
	nvcc -ccbin=$(CXX) -arch=$(ARCH) write.cu -o write
# remember to add optimization later on
clean:
	rm -f write
