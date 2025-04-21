// sample_fields.cu
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "dipole_field.hpp"
#include "load_dipole_data.hpp"

__device__ cuDoubleComplex complex_conj(cuDoubleComplex z) {
    return make_cuDoubleComplex(cuCreal(z), -cuCimag(z));
}

__device__ double real_cross_dot(const cvec3& E, const cvec3& B) {
    // Real part of (E x B*) . ẑ
    cuDoubleComplex ExBy = cuCmul(E.x, complex_conj(B.y));
    cuDoubleComplex EyBx = cuCmul(E.y, complex_conj(B.x));
    cuDoubleComplex ExBz = cuCmul(E.x, complex_conj(B.z));
    cuDoubleComplex EzBx = cuCmul(E.z, complex_conj(B.x));
    cuDoubleComplex EyBz = cuCmul(E.y, complex_conj(B.z));
    cuDoubleComplex EzBy = cuCmul(E.z, complex_conj(B.y));

    cuDoubleComplex S_z = cuCsub(ExBy, EyBx); // (E x B*) . z
    return cuCreal(S_z);
}

__global__ void compute_poynting_flux(const cvec3* E, const cvec3* B, double* S_out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double mu0 = 1.25663706212e-6;
    S_out[i] = 0.5 * real_cross_dot(E[i], B[i]) / mu0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " dipole_data.csv frequency_in_Hz" << std::endl;
        return 1;
    }

    // Load dipole data from CSV
    std::vector<vec3> host_positions;
    std::vector<cvec3> host_dipoles;
    load_dipole_data(argv[1], host_positions, host_dipoles);

    int N_dip = host_positions.size();

    // Define sampling grid (example: 100x100 grid at z = -100nm)
    const int Nx = 300, Ny = 300;
    const double z_sample = 6000e-9;
    const double grid_size = 30e-6; // 30 micron patch
    const double dx = grid_size / (Nx - 1);
    const double dy = grid_size / (Ny - 1);

    std::vector<vec3> host_obs(Nx * Ny);
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            double x = -grid_size/2 + dx * ix;
            double y = -grid_size/2 + dy * iy;
            host_obs[iy * Nx + ix] = {x, y, z_sample};
        }
    }

    int N_obs = Nx * Ny;
    std::vector<cvec3> host_E(N_obs);
    std::vector<cvec3> host_B(N_obs);
    std::vector<double> host_S(N_obs);

    // Allocate and copy data to device
    vec3* d_positions;
    cvec3* d_dipoles;
    vec3* d_obs;
    cvec3* d_E;
    cvec3* d_B;
    double* d_S;
    cudaMalloc(&d_positions, N_dip * sizeof(vec3));
    cudaMalloc(&d_dipoles, N_dip * sizeof(cvec3));
    cudaMalloc(&d_obs, N_obs * sizeof(vec3));
    cudaMalloc(&d_E, N_obs * sizeof(cvec3));
    cudaMalloc(&d_B, N_obs * sizeof(cvec3));
    cudaMalloc(&d_S, N_obs * sizeof(double));

    cudaMemcpy(d_positions, host_positions.data(), N_dip * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dipoles, host_dipoles.data(), N_dip * sizeof(cvec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obs, host_obs.data(), N_obs * sizeof(vec3), cudaMemcpyHostToDevice);

    // Set physical parameters
    // Read frequency from command line
    double frequency = std::atof(argv[2]);
    double c = 299792458.0; // speed of light in vacuum
    double lambda = c / frequency;
    double k = 2 * M_PI / lambda;
    double prefac = 1.0 / (4 * M_PI * 8.854187817e-12); // 1/(4pi*epsilon0)

    // Launch field kernel
    dim3 blockSize(256);
    dim3 gridSize((N_obs + blockSize.x - 1) / blockSize.x);
    compute_field<<<gridSize, blockSize>>>(d_positions, d_dipoles, N_dip, d_obs, d_E, d_B, N_obs, k, prefac);
    cudaDeviceSynchronize();

    // Launch Poynting kernel
    compute_poynting_flux<<<gridSize, blockSize>>>(d_E, d_B, d_S, N_obs);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(host_E.data(), d_E, N_obs * sizeof(cvec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B.data(), d_B, N_obs * sizeof(cvec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_S.data(), d_S, N_obs * sizeof(double), cudaMemcpyDeviceToHost);

    // Integrate Poynting flux
    double total_flux = 0.0;
    for (int i = 0; i < N_obs; ++i) {
        total_flux += host_S[i];
    }
    total_flux *= dx * dy;

    std::cout << "(" << frequency << "," << total_flux << ")," << std::endl;
    // std::cout << "Total power transmitted through plane: " << total_flux << " W" << std::endl;
    // std::cout << "Total radiated power estimate: " << total_flux*2 << " W" << std::endl;

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_dipoles);
    cudaFree(d_obs);
    cudaFree(d_E);
    cudaFree(d_B);
    cudaFree(d_S);

    return 0;
}
