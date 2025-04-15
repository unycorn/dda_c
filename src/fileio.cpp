#include "fileio.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>      // for std::exit
#include <complex>

#include <cuComplex.h>  // for cuDoubleComplex
#include <cuda_runtime.h>

cuDoubleComplex* load_matrix(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: could not open " << filename << " for reading.\n";
        std::exit(EXIT_FAILURE);
    }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N * N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N * N);
    in.close();
    return data;
}

cuDoubleComplex* load_vector(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: could not open " << filename << " for reading.\n";
        std::exit(EXIT_FAILURE);
    }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N);
    in.close();
    return data;
}

void write_polarizations(
    const char* filename,
    std::complex<double>* p, 
    std::vector<vec3> positions, 
    std::complex<double> alpha, 
    std::vector<std::complex<double>> E_inc, 
    int N
    ) {

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    out << "Re_px,Im_px,Re_py,Im_py,Re_pz,Im_pz,x,y,z,Re_alpha,Im_alpha,Re_Ex,Im_Ex,Re_Ey,Im_Ey,Re_Ez,Im_Ez\n";
    out << std::scientific << std::setprecision(9);

    for (int j = 0; j < N; ++j) {
        int idx = 3 * j;
        out
            << p[idx + 0].real() << ',' << p[idx + 0].imag() << ','
            << p[idx + 1].real() << ',' << p[idx + 1].imag() << ','
            << p[idx + 2].real() << ',' << p[idx + 2].imag() << ',';
        out
            << positions[j].x << ','
            << positions[j].y << ','
            << positions[j].z << ',';
        out
            << alpha.real() << ','
            << alpha.imag() << ',';
        out
            << E_inc[idx + 0].real() << ',' << E_inc[idx + 0].imag() << ','
            << E_inc[idx + 1].real() << ',' << E_inc[idx + 1].imag() << ','
            << E_inc[idx + 2].real() << ',' << E_inc[idx + 2].imag() << '\n';
    }

    out.close();
}
