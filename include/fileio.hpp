#ifndef FILEIO_HPP
#define FILEIO_HPP

#include <cuComplex.h>

// Reads a square complex matrix from a binary file.
// First 4 bytes: int (N), then N*N cuDoubleComplex elements.
cuDoubleComplex* load_matrix(const char* filename, int& N);

// Reads a complex vector from a binary file.
// First 4 bytes: int (N), then N cuDoubleComplex elements.
cuDoubleComplex* load_vector(const char* filename, int& N);

// Writes polarizations (3N-vector of cuDoubleComplex) to CSV text file.
void write_polarizations(const char* filename, const cuDoubleComplex* p, int N);

#endif // FILEIO_HPP
