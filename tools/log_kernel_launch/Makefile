# -------- edit ONLY the next line if your NVBit tree lives elsewhere -------
NVBIT_PATH := /home/vic/Dev/nvbit-1.7/core   # <‑‑ absolute path to NVBit/core
# --------------------------------------------------------------------------

CXX   = /usr/bin/g++-13
NVCC  = nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS = ptxas

# ---------- CUDA / PTXAS version checks (unchanged) ------------------------
NVCC_VER_REQ = 10.1
NVCC_VER     = $(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_OK      = $(shell echo "$(NVCC_VER) >= $(NVCC_VER_REQ)" | bc)
ifeq ($(NVCC_OK),0)
$(error nvcc >= $(NVCC_VER_REQ) required)
endif

PTXAS_VER_ADD_FLAG = 12.3
PTXAS_VER     = $(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_OK      = $(shell echo "$(PTXAS_VER) >= $(PTXAS_VER_ADD_FLAG)" | bc)
ifeq ($(PTXAS_OK),0)
MAXRREGCOUNT_FLAG = -maxrregcount=24
else
MAXRREGCOUNT_FLAG =
endif

# ---------- include / link paths ------------------------------------------
CUDA_INC  := /usr/local/cuda/include
INCLUDES  := -I$(NVBIT_PATH) -I$(CUDA_INC)
LIBS      := -L$(NVBIT_PATH) -lnvbit
NVCC_PATH := -L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# ---------- files ---------------------------------------------------------
SRC      := log_kernel_launch.cu
OBJ      := log_kernel_launch.o
SO       := log_kernel_launch.so
ARCH    ?= all

DBGFLAG := -DDEBUG=1

# ---------- rules ---------------------------------------------------------
all: $(SO)

$(SO): $(OBJ) /home/vic/Dev/nvbit-1.7/core/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(DBGFLAG) -O3 -shared $(OBJ) $(LIBS) $(NVCC_PATH) \
	        -lcuda -lcudart_static  -o $@
$(OBJ): $(SRC) cuda_v1_shim.h
	$(NVCC) -dc -c -std=c++17 $(DBGFLAG) $(INCLUDES) -Xptxas -cloning=no \
	        -Xcompiler -Wall -arch=$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

clean:
	rm -f *.o *.soINCLUDES=-I$(NVBIT_PATH) -I$(NVBIT_PATH)/core -I/usr/local/cuda/include