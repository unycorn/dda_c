#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <complex>
#include <sstream>
#include <cuda_runtime.h>

#include "constants.hpp"
#include "vector3.hpp"
#include "interaction.hpp"
#include "fileio.hpp"
#include "solve_gpu.hpp"
#include "print_utils.hpp"
#include "lorentzian.hpp"
#include "matrix_ops.hpp"
#include "rotation.hpp"

using mat2x2 = std::complex<double>[2][2];
constexpr std::complex<double> I(0.0, 1.0);

// ---- Main Simulation Function ----
void run_simulation(
    double f_start,
    double f_end,
    int num_freqs,
    const std::vector<vec3>& positions,
    const std::vector<LorentzianParams>& params_00,
    const std::vector<LorentzianParams>& params_05,
    const std::vector<LorentzianParams>& params_50,
    const std::vector<LorentzianParams>& params_55,
    const std::vector<double>& angles
) {
    const int N = positions.size();
    for (int i = 0; i < num_freqs; ++i) {
        double freq = (num_freqs == 1) ? f_start : f_start + i * (f_end - f_start) / (num_freqs - 1);
        
        // Check if output file already exists
        std::ostringstream filename;
        filename << "output/output_freq_" << std::scientific << std::setprecision(2) << freq << ".csv";
        
        if (std::ifstream(filename.str()).good()) {
            std::cout << "Skipping frequency " << freq << " Hz - output file already exists\n";
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        std::vector<mat2x2> alpha(N);
        
        // Generate polarizability matrix using the provided parameters for each dipole
        // This is 2x2 and scalar because it couples only e"x" and m"z" dipoles to fields
        for (int j = 0; j < N; ++j) {
            alpha[j][0][0] = lorentz_alpha_params(freq, params_00[j]);
            alpha[j][0][1] = lorentz_alpha_params(freq, params_05[j]);
            alpha[j][1][0] = lorentz_alpha_params(freq, params_50[j]);
            alpha[j][1][1] = lorentz_alpha_params(freq, params_55[j]);
        }

        std::vector<std::complex<double>> A_host(2 * N * 2 * N, std::complex<double>(0.0, 0.0));
        cuDoubleComplex* A_device = get_full_interaction_matrix_scalar(A_host.data(), positions.data(), alpha.data(), angles.data(), N, k);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        try {
            // Initialize incident field (2N components - one electric and one magnetic scalar per dipole)
            std::vector<std::complex<double>> inc_field(2 * N, std::complex<double>(0.0, 0.0));
            for (int j = 0; j < N; ++j) {
                // Calculate phase at dipole position (these should all be identity)
                double phase = k * positions[j].z;
                std::complex<double> phase_factor = std::exp(I * phase);
                
                // Create complex vectors for E and H fields
                vec3 E_inc_real = {1.0, 0.0, 0.0};  // Real part of E-field (x-component)
                vec3 H_inc_real = {0.0, 1.0/Z_0, 0.0};  // Real part of H-field (y-component)
                
                // Create projection vectors
                vec3 u_e = {cos(angles[j]), sin(angles[j]), 0.0}; // Electric projection vector
                vec3 u_h = {0.0, 0.0, 1.0}; // Magnetic projection vector (assumed along z-axis)
                
                // Calculate projections and multiply by phase factor
                inc_field[2 * j + 0] = phase_factor * vec3_dot(E_inc_real, u_e);     // Electric projection
                inc_field[2 * j + 1] = phase_factor * vec3_dot(H_inc_real, u_h);     // Magnetic projection
            }

            // Convert incident field to cuDoubleComplex for GPU
            std::vector<cuDoubleComplex> b_cuda(2 * N);
            for (int i = 0; i < 2 * N; ++i) {
                b_cuda[i] = make_cuDoubleComplex(std::real(inc_field[i]), std::imag(inc_field[i]));
            }

            // Transpose matrix for GPU solver (expects column-major format)
            std::vector<cuDoubleComplex> A_transposed(2 * N * 2 * N);
            for (int i = 0; i < 2 * N; ++i) {
                for (int j = 0; j < 2 * N; ++j) {
                    A_transposed[i * 2 * N + j] = make_cuDoubleComplex(
                        std::real(A_host[j * 2 * N + i]),
                        std::imag(A_host[j * 2 * N + i])
                    );
                }
            }
            
            // print_complex_matrix("Transposed Interaction Matrix", A_transposed.data(), 2 * N);

            cudaMemcpy(A_device, A_transposed.data(), sizeof(cuDoubleComplex) * 2 * N * 2 * N, cudaMemcpyHostToDevice);
            
            solve_gpu(A_device, b_cuda.data(), 2 * N);

            std::vector<std::complex<double>> b(2 * N);
            for (int i = 0; i < 2 * N; ++i) {
                b[i] = std::complex<double>(cuCreal(b_cuda[i]), cuCimag(b_cuda[i]));
            }

            if (A_device != nullptr) {
                cudaFree(A_device);
                A_device = nullptr;
            }

            write_polarizations(filename.str().c_str(), b.data(), positions, alpha, N);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Elapsed: " << ms_duration.count() << " ms\n";

        } catch (const std::exception& e) {
            std::cerr << "Error during frequency " << freq << ": " << e.what() << std::endl;
            if (A_device != nullptr) {
                cudaFree(A_device);
                A_device = nullptr;
            }
            throw;
        }
    }
}


// ---- Main Function ----
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_csv_file>\n";
        std::cerr << "  input_csv_file: Path to CSV file containing dipole parameters\n";
        std::cerr << "  CSV format: x,y,z,f0_00,gamma_00,A_00,...,angle\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::vector<vec3> positions;
    std::vector<LorentzianParams> params_00, params_05, params_50, params_55;
    std::vector<double> angles;

    // Load the dipole parameters from CSV
    std::ifstream file(input_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file " << input_file << std::endl;
        return 1;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;
        
        while (std::getline(ss, value, ',')) {
            values.push_back(std::stod(value));
        }

        // Expected CSV format:
        // x,y,z,f0_00,gamma_00,A_00,B_00,C_00,f0_05,gamma_05,A_05,B_05,C_05,f0_50,gamma_50,A_50,B_50,C_50,f0_55,gamma_55,A_55,B_55,C_55,angle

        positions.push_back(vec3{values[0], values[1], values[2]});
        angles.push_back(values[3]);
        std::cout << values[0] << "\n";

        // Create LorentzianParams for each component
        params_00.push_back(LorentzianParams{
            values[6], values[7], values[8],  // A, B, C
            values[4], values[5],             // F0, gamma
            LorentzianForm::STANDARD
        });

        params_05.push_back(LorentzianParams{
            values[11], values[12], values[13],  // A, B, C
            values[9], values[10],               // F0, gamma
            LorentzianForm::IF_NUMERATOR
        });

        params_50.push_back(LorentzianParams{
            values[16], values[17], values[18],  // A, B, C
            values[14], values[15],              // F0, gamma
            LorentzianForm::NEG_IF_NUMERATOR
        });

        params_55.push_back(LorentzianParams{
            values[21], values[22], values[23],  // A, B, C
            values[19], values[20],              // F0, gamma
            LorentzianForm::F2_NUMERATOR
        });
    }

    int N = positions.size();
    if (N == 0) {
        std::cerr << "Error: No dipoles loaded from input file\n";
        return 1;
    }

    // Run the simulation with the loaded parameters
    run_simulation(150e12, 350e12, 50, positions, params_00, params_05, params_50, params_55, angles);

    // Test biani_green_matrix_scalar with two test points
    // vec3 point1 = {0.0, 0.0, 0.0};  // Origin
    // vec3 point2 = {1e-6, 0.0, 0.0};  // 1 micron away in x direction
    // double angle1 = 0.0;  // No rotation
    // double angle2 = M_PI/4.0;  // 45 degree rotation
    // double k = 2.0 * M_PI / (C_LIGHT / 220e12);  // Wavevector at 220 THz
    
    // std::complex<double> result[4];  // Flat array for 2x2 matrix
    // biani_green_matrix_scalar(
    //     result, point1, point2, angle1, angle2, k
    // );
    
    // // Print the resulting 2x2 matrix
    // std::cout << "\nTesting biani_green_matrix_scalar:\n";
    // std::cout << "Points: (0,0,0) and (1μm,0,0)\n";
    // std::cout << "Angles: 0° and 45°\n";
    // std::cout << "Matrix result:\n";
    // std::cout << result[0] << "  " << result[1] << "\n";
    // std::cout << result[2] << "  " << result[3] << "\n";

    return 0;
}
