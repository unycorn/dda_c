#include <iostream>
#include <fstream>
#include <cuComplex.h>

cuDoubleComplex* load_matrix(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << filename << "\n"; std::exit(1); }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N * N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N * N);
    in.close();
    return data;
}

cuDoubleComplex* load_vector(const char* filename, int& N) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) { std::cerr << "Failed to open " << filename << "\n"; std::exit(1); }

    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    cuDoubleComplex* data = new cuDoubleComplex[N];
    in.read(reinterpret_cast<char*>(data), sizeof(cuDoubleComplex) * N);
    in.close();
    return data;
}

void write_polarizations(const char* filename, cuDoubleComplex* p, int N) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: could not open " << filename << " for writing\n";
        std::exit(1);
    }

    // CSV header
    out << "Re_px,Im_px,Re_py,Im_py,Re_pz,Im_pz\n";

    out << std::scientific << std::setprecision(6);
    for (int j = 0; j < N; ++j) {
        int idx = 3 * j;
        out
            << cuCreal(p[idx + 0]) << ',' << cuCimag(p[idx + 0]) << ','  // x
            << cuCreal(p[idx + 1]) << ',' << cuCimag(p[idx + 1]) << ','  // y
            << cuCreal(p[idx + 2]) << ',' << cuCimag(p[idx + 2]) << '\n'; // z
    }

    out.close();
}