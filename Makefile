# Compiler settings
CXX := g++
CXXFLAGS := -std=c++11 -O3 -Wall -Iinclude -fopenmp
LDFLAGS := -fopenmp

# CUDA paths - make configurable for different systems
CUDA_PATH ?= /usr/local/cuda
LDFLAGS += -L$(CUDA_PATH)/lib64 -L/usr/lib64 -lcudart -lcusolver -lcublas -llapack -lblas
CXXFLAGS += -I$(CUDA_PATH)/include -I/opt/cuda/include -I/usr/local/cuda/include

# Google Test settings - build from source
GTEST_VERSION := release-1.11.0
GTEST_DIR := third_party/googletest
GTEST_ZIP := third_party/googletest-$(GTEST_VERSION).zip
GTEST_URL := https://github.com/google/googletest/archive/refs/tags/$(GTEST_VERSION).zip
GTEST_BUILD := $(GTEST_DIR)/build
GTEST_LIBS := $(GTEST_BUILD)/lib/libgtest.a $(GTEST_BUILD)/lib/libgtest_main.a

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

# Download and build Google Test
$(GTEST_ZIP):
	mkdir -p third_party
	curl -L $(GTEST_URL) -o $(GTEST_ZIP)

$(GTEST_DIR): $(GTEST_ZIP)
	cd third_party && unzip googletest-$(GTEST_VERSION).zip
	mv third_party/googletest-$(GTEST_VERSION) $(GTEST_DIR)

$(GTEST_LIBS): $(GTEST_DIR)
	mkdir -p $(GTEST_BUILD)
	cd $(GTEST_BUILD) && cmake .. -DCMAKE_CXX_FLAGS="-fPIC" && make

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

# Deep clean (including Google Test)
distclean: clean
	rm -rf third_party

.PHONY: all clean distclean test
