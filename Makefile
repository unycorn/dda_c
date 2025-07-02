# Compiler settings
CXX := g++
CXXFLAGS := -std=c++11 -O3 -Wall -Iinclude -fopenmp
LDFLAGS := -fopenmp

# CUDA paths - make configurable for different systems
CUDA_PATH ?= /usr/local/cuda
LDFLAGS += -L$(CUDA_PATH)/lib64 -L/usr/lib64 -lcudart -lcusolver -lcublas -l:liblapack.so.3 -l:libblas.so.3
CXXFLAGS += -I$(CUDA_PATH)/include -I/opt/cuda/include -I/usr/local/cuda/include

# Directories
SRC_DIR := src
OBJ_DIR := obj
TEST_DIR := tests
TEST_BIN := run_tests
BIN := solver

# Source and object files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# Test source and object files (excluding main.cpp)
TEST_SRCS := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(TEST_SRCS))
COMMON_OBJS := $(filter-out $(OBJ_DIR)/main.o,$(OBJS))

# Default target
all: $(BIN)

# Link main program
$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Link test program with static Google Test libraries
$(TEST_BIN): $(TEST_OBJS) $(COMMON_OBJS) $(GTEST_LIBS)
	$(CXX) $(CXXFLAGS) $(TEST_OBJS) $(COMMON_OBJS) -o $@ $(LDFLAGS) $(GTEST_LIBS) -lpthread

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile test files with Google Test include path
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(OBJ_DIR) $(GTEST_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR)/include -c $< -o $@

# Create object directory
$(OBJ_DIR):
	mkdir -p $@

# Test target
test: $(TEST_BIN)
	./$(TEST_BIN)

# Clean more thoroughly
clean:
	rm -rf $(OBJ_DIR) $(BIN) $(TEST_BIN) third_party

.PHONY: all clean test
