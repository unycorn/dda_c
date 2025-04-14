CXX = g++
CXXFLAGS = -O2 -std=c++11 -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lcusolver -lcublas

TARGET = solver
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
