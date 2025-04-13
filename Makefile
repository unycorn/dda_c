# Name of your final executable
TARGET = solver

# Source files
SRC = parallel_lapack_solver.c

# Compiler and flags
CC = cc
CFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)
