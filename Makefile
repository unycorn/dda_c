# Detect OS
UNAME_S := $(shell uname -s)

# Source and object files
SRC = main.c interaction.c vector3.c
OBJ = $(SRC:.c=.o)

# Output binary name
TARGET = dda_solver

# Compiler and flags
ifeq ($(UNAME_S), Darwin)
    # macOS (Apple Clang has no OpenMP support)
    CC = /opt/homebrew/bin/gcc-14   # or whatever version is installed
    CFLAGS = -std=c11 -O2 -fopenmp
    LDFLAGS = -L/opt/homebrew/opt/libomp/lib -Wl,-rpath,/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate
else
    # Linux: Use default gcc
    CC = gcc
    CFLAGS = -std=c11 -O2 -fopenmp
    LDFLAGS = -L$(HOME)/openblas/lib -Wl,-rpath,$(HOME)/openblas/lib -lopenblas -lm
	CFLAGS += -I$(HOME)/openblas/include

endif

# Default build rule
all: $(TARGET)

# Link final binary
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDFLAGS) -o $@

# Compile each .c file into .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJ) $(TARGET)
