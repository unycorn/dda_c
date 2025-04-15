# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Iinclude

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
	$(CXX) $(CXXFLAGS) -o $@ $^

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