# Makefile for compiling dipole field sampling CUDA program

# Compiler and flags
NVCC := nvcc
CXXFLAGS := -std=c++11
NVCCFLAGS := -arch=sm_60 -std=c++11

# Source files
CU_SOURCES := sample_fields.cu dipole_field.cu
CPP_SOURCES := load_dipole_data.cpp
EXTRA_OBJS := ../src/fileio.o

# Object files
CU_OBJS := $(CU_SOURCES:.cu=.o)
CPP_OBJS := $(CPP_SOURCES:.cpp=.o)

# Executable
TARGET := sample_fields

all: $(TARGET)

$(TARGET): $(CU_OBJS) $(CPP_OBJS) $(EXTRA_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(TARGET)
