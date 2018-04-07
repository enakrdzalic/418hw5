# Source files
BIN = conv
CC_SRC := $(wildcard *.c)
CPP_SRC := $(wildcard *.cpp)
CU_SRC := $(wildcard *.cu)

# Use either GetTimeOfDay or RUsage.
TIMING = GetTimeOfDay
#TIMING = RUsage

# Compiler and linker options
INC_DIR= 
CXX=g++
CXXFLAGS= -O3 -m64 -Wall -std=c++11 #-g

NVCC      = /cs/local/lib/pkg/cudatoolkit/bin/nvcc
# The GTX 1060s are CC 6.1 (Pascal), but the highest level supported
# by our current CUDA toolkit (version 7.5) is CC 5.x (Maxwell) so we
# choose 5.2 as the highest which had GTX models.
NVCCFLAGS =-O3 -m64 -arch compute_52

L_DIR = -L/cs/local/lib/pkg/cudatoolkit/lib64
LIBS = cudart #cublas
LDLIBS  := $(addprefix -l, $(LIBS))

# Build folder
BUILD_DIR=build
CC_OBJS=$(CC_SRC:%.c=$(BUILD_DIR)/%.o)
CPP_OBJS=$(CPP_SRC:%.cpp=$(BUILD_DIR)/%.o)
CU_OBJS=$(CU_SRC:%.cu=$(BUILD_DIR)/%.o)

default: $(BIN)
dirs:
	mkdir -p $(BUILD_DIR)/
clean:
	rm -rf $(BUILD_DIR) *~ $(BIN)

$(BIN): dirs $(CU_OBJS) $(CC_OBJS) $(CPP_OBJS)
#	$(NVCC) $(NVCCFLAGS) -o $@ $(CU_OBJS) $(CC_OBJS) $(CPP_OBJS) $(L_DIR) $(LDLIBS)
	$(CXX) $(CXXFLAGS) -o $@ $(CU_OBJS) $(CC_OBJS) $(CPP_OBJS) $(L_DIR) $(LDLIBS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) $< $(INC_DIR) $(CXXFLAGS) -D $(TIMING) -c -o $@

$(BUILD_DIR)/%.o: %.c
	$(CXX) $< $(INC_DIR) $(CXXFLAGS) -D $(TIMING) -c -o $@

$(BUILD_DIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -D $(TIMING) -c -o $@
