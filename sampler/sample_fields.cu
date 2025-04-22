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

    // Define sampling grid (example: 300x300 grid at z = 1000nm)
    const int Nx = 100, Ny = 100;
    const double z_sample = 5000e-9;
    const double center_x = 5e-6;
    const double center_y = 5e-6;
    const double grid_size = 9e-6; // 10 micron patch
    const double dx = grid_size / (Nx - 1);
    const double dy = grid_size / (Ny - 1);

    std::vector<vec3> host_obs(Nx * Ny);
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            double x = center_x + dx * (ix - Nx/2);
            double y = center_y + dy * (iy - Ny/2);
            host_obs[iy * Nx + ix] = {x, y, z_sample};

            // std::cout << x << " " << y << " " << z_sample << std::endl;
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
    double Z0 = 376.73; // Impedance of free space

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
        // std::cout << host_S[i] << std::endl;
    }
    double avg_flux = total_flux / N_obs;
    total_flux *= dx * dy;

    std::cout << "(" << frequency << "," << avg_flux*2.0*Z0 << ")," << std::endl;

    // Add single point calculation
    // vec3 single_point = {0.0, 0.0, 1e-6};  // Point of interest
    // std::vector<vec3> single_obs(1, single_point);
    // std::vector<cvec3> single_E(1);
    // std::vector<cvec3> single_B(1);

    // // Allocate device memory for single point
    // vec3* d_single_obs;
    // cvec3* d_single_E;
    // cvec3* d_single_B;
    // cudaMalloc(&d_single_obs, sizeof(vec3));
    // cudaMalloc(&d_single_E, sizeof(cvec3));
    // cudaMalloc(&d_single_B, sizeof(cvec3));

    // // Copy single observation point to device
    // cudaMemcpy(d_single_obs, single_obs.data(), sizeof(vec3), cudaMemcpyHostToDevice);

    // // Compute fields at single point
    // compute_field<<<1, 1>>>(d_positions, d_dipoles, N_dip, d_single_obs, d_single_E, d_single_B, 1, k, prefac);
    // cudaDeviceSynchronize();

    // // Copy results back
    // cudaMemcpy(single_E.data(), d_single_E, sizeof(cvec3), cudaMemcpyDeviceToHost);
    // cudaMemcpy(single_B.data(), d_single_B, sizeof(cvec3), cudaMemcpyDeviceToHost);

    // // Print single point results
    // std::cout << "\nFields at point (0, 0, 1e-6):" << std::endl;
    // std::cout << "E-field (V/m): (" 
    //           << cuCreal(single_E[0].x) << " + " << cuCimag(single_E[0].x) << "i, "
    //           << cuCreal(single_E[0].y) << " + " << cuCimag(single_E[0].y) << "i, "
    //           << cuCreal(single_E[0].z) << " + " << cuCimag(single_E[0].z) << "i)" << std::endl;
    // std::cout << "B-field (T): (" 
    //           << cuCreal(single_B[0].x) << " + " << cuCimag(single_B[0].x) << "i, "
    //           << cuCreal(single_B[0].y) << " + " << cuCimag(single_B[0].y) << "i, "
    //           << cuCreal(single_B[0].z) << " + " << cuCimag(single_B[0].z) << "i)" << std::endl;

    // // Free additional device memory
    // cudaFree(d_single_obs);
    // cudaFree(d_single_E);
    // cudaFree(d_single_B);

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_dipoles);
    cudaFree(d_obs);
    cudaFree(d_E);
    cudaFree(d_B);
    cudaFree(d_S);

    return 0;
}
