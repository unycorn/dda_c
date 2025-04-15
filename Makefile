# Compiler settings
CXX := g++
CXXFLAGS := -std=c++11 -O3 -Wall -Iinclude

# CUDA paths (update if different)
CUDA_PATH := /usr/local/cuda
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lcusolver
CXXFLAGS += -I$(CUDA_PATH)/include

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN := solver

# Source and object files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# Default target
all: $(BIN)

# Link
$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Compile
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create object directory
$(OBJ_DIR):
	mkdir -p $@

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN)

.PHONY: all clean
