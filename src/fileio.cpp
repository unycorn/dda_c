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
    
    out << std::scientific << std::setprecision(4);

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

void write_polarizations_binary(
    const char* filename,
    std::complex<double>* p, 
    std::vector<vec3> positions,
    const std::vector<std::complex<double>[2][2]>& alpha,
    int N,
    double frequency
) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    // Write the size N first
    out.write(reinterpret_cast<const char*>(&N), sizeof(int));
    
    // Write the frequency
    out.write(reinterpret_cast<const char*>(&frequency), sizeof(double));

    // Write the polarization data
    for (int n = 0; n < N; ++n) {
        // Write Ex,Mz components
        out.write(reinterpret_cast<const char*>(&p[2*n + 0]), sizeof(std::complex<double>));     // Ex
        out.write(reinterpret_cast<const char*>(&p[2*n + 1]), sizeof(std::complex<double>));     // Mz
    }

    out.close();
}

void write_PW_sweep_polarization_binary(
    const char* filename,
    const std::vector<std::complex<double>>& polarizations,
    const std::vector<vec3>& positions,
    const std::vector<std::complex<double>[2][2]>& alpha,
    const std::vector<double>& kx_values,
    const std::vector<double>& ky_values,
    const std::vector<std::string>& polarization_types,
    int N,
    double frequency
) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing.\n";
        std::exit(EXIT_FAILURE);
    }

    // Write metadata
    int num_k_points = kx_values.size();
    int num_polarizations = polarization_types.size();
    
    out.write(reinterpret_cast<const char*>(&N), sizeof(int));
    out.write(reinterpret_cast<const char*>(&frequency), sizeof(double));
    out.write(reinterpret_cast<const char*>(&num_k_points), sizeof(int));
    out.write(reinterpret_cast<const char*>(&num_polarizations), sizeof(int));
    
    // Write k-vector data
    for (int i = 0; i < num_k_points; ++i) {
        out.write(reinterpret_cast<const char*>(&kx_values[i]), sizeof(double));
        out.write(reinterpret_cast<const char*>(&ky_values[i]), sizeof(double));
    }
    
    // Write polarization type strings
    for (int i = 0; i < num_polarizations; ++i) {
        int str_len = polarization_types[i].length();
        out.write(reinterpret_cast<const char*>(&str_len), sizeof(int));
        out.write(polarization_types[i].c_str(), str_len);
    }
    
    // Write polarization data: [N_dipoles * 2 * N_k_points * N_polarizations]
    int total_elements = 2 * N * num_k_points * num_polarizations;
    for (int i = 0; i < total_elements; ++i) {
        out.write(reinterpret_cast<const char*>(&polarizations[i]), sizeof(std::complex<double>));
    }

    out.close();
}

std::vector<std::complex<double>> read_polarizations_binary(const char* filename, int& N, double& frequency) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Error: could not open " << filename << " for reading.\n";
        std::exit(EXIT_FAILURE);
    }

    // Read the size N first
    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    
    // Read the frequency
    in.read(reinterpret_cast<char*>(&frequency), sizeof(double));

    // Each point has 2 complex values (Ex and Mz)
    std::vector<std::complex<double>> p(2 * N);
    
    // Read all the polarization data
    in.read(reinterpret_cast<char*>(p.data()), sizeof(std::complex<double>) * 2 * N);

    in.close();
    return p;
}
