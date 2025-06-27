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
    const std::vector<std::complex<double>[6][6]>& alpha,
    int N
    ) {

    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    // Write header
    out << "Re_px,Im_px,Re_py,Im_py,Re_pz,Im_pz,Re_mx,Im_mx,Re_my,Im_my,Re_mz,Im_mz,x,y,z";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            out << ",Re_alpha" << i << j << ",Im_alpha" << i << j;
        }
    }
    out << "\n";
    
    out << std::scientific << std::setprecision(9);

    for (int n = 0; n < N; ++n) {
        // Write all 6 components of p (3 electric + 3 magnetic)
        for (int j = 0; j < 6; ++j) {
            if (j > 0) out << ",";
            out << p[6*n + j].real() << "," << p[6*n + j].imag();
        }
        
        // Write positions
        out << "," << positions[n].x << ","
            << positions[n].y << ","
            << positions[n].z;
        
        // Write all 36 elements of the alpha matrix
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                out << "," << alpha[n][i][j].real() << "," << alpha[n][i][j].imag();
            }
        }
        out << "\n";
    }

    out.close();
}

void write_polarizations(
    const char* filename,
    std::complex<double>* p, 
    std::vector<vec3> positions,
    const std::vector<std::complex<double>[2][2]>& alpha,
    int N
) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    // Write header for 2x2 case (only Ex,Mz components)
    out << "Re_px,Im_px,Re_mz,Im_mz";
    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < 2; ++j) {
    //         out << ",Re_alpha" << i << j << ",Im_alpha" << i << j;
    //     }
    // }
    out << "\n";
    
    out << std::scientific << std::setprecision(5);

    // Write data
    for (int n = 0; n < N; ++n) {
        // Write Ex,Mz components only
        out << p[2*n + 0].real() << "," << p[2*n + 0].imag() << ",";     // Ex
        out << p[2*n + 1].real() << "," << p[2*n + 1].imag();            // Mz
        
        // // Write positions
        // out << "," << positions[n].x << ","
        //     << positions[n].y << ","
        //     << positions[n].z;
        
        // Write all 4 elements of the alpha matrix
        // for (int i = 0; i < 2; ++i) {
        //     for (int j = 0; j < 2; ++j) {
        //         out << "," << alpha[n][i][j].real() << "," << alpha[n][i][j].imag();
        //     }
        // }
        out << "\n";
    }

    out.close();
}
