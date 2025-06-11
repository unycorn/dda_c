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

__global__ void combine_fields(cvec3* E_total, cvec3* B_total, 
                            const cvec3* E_electric, const cvec3* B_electric,
                            const cvec3* E_magnetic, const cvec3* B_magnetic,
                            int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double eps0 = 8.854187817e-12; // Vacuum permittivity
    double mu0 = 1.25663706212e-6; // Vacuum permeability
    double scale = eps0 * mu0;

    // Combine E fields via duality
    E_total[i].x = cuCadd(E_electric[i].x, cuCmul(make_cuDoubleComplex(-1, 0), B_magnetic[i].x));
    E_total[i].y = cuCadd(E_electric[i].y, cuCmul(make_cuDoubleComplex(-1, 0), B_magnetic[i].y));
    E_total[i].z = cuCadd(E_electric[i].z, cuCmul(make_cuDoubleComplex(-1, 0), B_magnetic[i].z));

    // Combine B fields duality
    B_total[i].x = cuCadd(B_electric[i].x, cuCmul(make_cuDoubleComplex(scale, 0), E_magnetic[i].x));
    B_total[i].y = cuCadd(B_electric[i].y, cuCmul(make_cuDoubleComplex(scale, 0), E_magnetic[i].y));
    B_total[i].z = cuCadd(B_electric[i].z, cuCmul(make_cuDoubleComplex(scale, 0), E_magnetic[i].z));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " angle_data.csv dipole_data.csv frequency_in_Hz" << std::endl;
        return 1;
    }

    // Load dipole data from CSV
    std::vector<vec3> host_positions;
    std::vector<cvec3> host_electric_dipoles;
    std::vector<cvec3> host_magnetic_dipoles;
    load_dipole_data(argv[1], argv[2], host_positions, host_electric_dipoles, host_magnetic_dipoles);

    int N_dip = host_positions.size();

    // Define sampling grid (example: 50x50 grid at z = 1000nm)
    const int Nx = 50, Ny = 50;
    const double z_sample = 1000e-9;
    const double center_x = Nx / 2.0 * 300e-9; // Center at 15 microns
    const double center_y = Ny / 2.0 * 300e-9; // Center at 15 microns
    const double grid_size = 300e-9 * (Nx - 2);
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
    std::vector<cvec3> host_E_electric(N_obs);
    std::vector<cvec3> host_B_electric(N_obs);
    std::vector<cvec3> host_E_magnetic(N_obs);
    std::vector<cvec3> host_B_magnetic(N_obs);
    std::vector<double> host_S(N_obs);

    // Allocate and copy data to device
    vec3* d_positions;
    cvec3* d_electric_dipoles;
    cvec3* d_magnetic_dipoles;
    vec3* d_obs;
    cvec3* d_E_electric;
    cvec3* d_B_electric;
    cvec3* d_E_magnetic;
    cvec3* d_B_magnetic;
    double* d_S;
    cudaMalloc(&d_positions, N_dip * sizeof(vec3));
    cudaMalloc(&d_electric_dipoles, N_dip * sizeof(cvec3));
    cudaMalloc(&d_magnetic_dipoles, N_dip * sizeof(cvec3));
    cudaMalloc(&d_obs, N_obs * sizeof(vec3));
    cudaMalloc(&d_E_electric, N_obs * sizeof(cvec3));
    cudaMalloc(&d_B_electric, N_obs * sizeof(cvec3));
    cudaMalloc(&d_E_magnetic, N_obs * sizeof(cvec3));
    cudaMalloc(&d_B_magnetic, N_obs * sizeof(cvec3));
    cudaMalloc(&d_S, N_obs * sizeof(double));

    cudaMemcpy(d_positions, host_positions.data(), N_dip * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_electric_dipoles, host_electric_dipoles.data(), N_dip * sizeof(cvec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_magnetic_dipoles, host_magnetic_dipoles.data(), N_dip * sizeof(cvec3), cudaMemcpyHostToDevice);
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
    compute_field<<<gridSize, blockSize>>>(d_positions, d_electric_dipoles, N_dip, d_obs, d_E_electric, d_B_electric, N_obs, k, prefac);
    compute_field<<<gridSize, blockSize>>>(d_positions, d_magnetic_dipoles, N_dip, d_obs, d_E_magnetic, d_B_magnetic, N_obs, k, prefac);
    cudaDeviceSynchronize();

    // Combine the fields
    combine_fields<<<gridSize, blockSize>>>(d_E_electric, d_B_electric, 
        d_E_electric, d_B_electric,
        d_E_magnetic, d_B_magnetic,
        N_obs);
    cudaDeviceSynchronize();

    // Now d_E_electric and d_B_electric contain the total fields
    // We can free d_E_magnetic and d_B_magnetic as they're no longer needed
    cudaFree(d_E_magnetic);
    cudaFree(d_B_magnetic);


    // Launch Poynting kernel
    compute_poynting_flux<<<gridSize, blockSize>>>(d_E_electric, d_B_electric, d_S, N_obs);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(host_E_electric.data(), d_E_electric, N_obs * sizeof(cvec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_B_electric.data(), d_B_electric, N_obs * sizeof(cvec3), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_S.data(), d_S, N_obs * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate reflection coefficient first (using just scattered fields)
    double total_flux_reflection = 0.0;
    for (int i = 0; i < N_obs; ++i) {
        total_flux_reflection += host_S[i];
    }
    double avg_flux_reflection = total_flux_reflection / N_obs;
    double total_flux_reflection_normalized = avg_flux_reflection * 2.0 * Z0;

    // Add incident plane wave fields for transmission calculation
    // E = x̂E₀exp(ikz), B = ŷ(E₀/c)exp(ikz)
    for (int i = 0; i < N_obs; ++i) {
        double phase = k * z_sample;
        cuDoubleComplex exp_ikz = make_cuDoubleComplex(cos(phase), sin(phase));
        
        // Add E field (x-polarized)
        host_E_electric[i].x = cuCadd(host_E_electric[i].x, exp_ikz);
        
        // Add B field (y-polarized, E₀/c amplitude for plane wave)
        host_B_electric[i].y = cuCadd(host_B_electric[i].y, make_cuDoubleComplex(
            cuCreal(exp_ikz) / c,
            cuCimag(exp_ikz) / c
        ));
    }

    // Recompute Poynting vector with total fields
    cudaMemcpy(d_E_electric, host_E_electric.data(), N_obs * sizeof(cvec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_electric, host_B_electric.data(), N_obs * sizeof(cvec3), cudaMemcpyHostToDevice);
    compute_poynting_flux<<<gridSize, blockSize>>>(d_E_electric, d_B_electric, d_S, N_obs);
    cudaDeviceSynchronize();
    cudaMemcpy(host_S.data(), d_S, N_obs * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate transmission coefficient
    double total_flux_transmission = 0.0;
    for (int i = 0; i < N_obs; ++i) {
        total_flux_transmission += host_S[i];
    }
    double avg_flux_transmission = total_flux_transmission / N_obs;
    double total_flux_transmission_normalized = avg_flux_transmission * 2.0 * Z0;

    // Print both reflection and transmission coefficients
    std::cout << "(" << frequency << "," << total_flux_reflection_normalized << "," 
              << total_flux_transmission_normalized << ")," << std::endl;

    // Free device memory
    cudaFree(d_positions);
    cudaFree(d_electric_dipoles);
    cudaFree(d_obs);
    cudaFree(d_E_electric);
    cudaFree(d_B_electric);
    cudaFree(d_S);

    return 0;
}
