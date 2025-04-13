# Compiler and flags
CC = cc
CFLAGS = -std=c11 -O2 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate

# Source files
SRC = main.c interaction.c vector3.c
OBJ = $(SRC:.c=.o)

# Executable name
TARGET = dda_solver

# Default rule
all: $(TARGET)

# Link the final binary
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $(TARGET)

# Compile each .c into .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJ) $(TARGET)
