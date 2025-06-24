#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <complex>
#include <sstream>
#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

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
    const std::vector<double>& angles,
    const std::string& output_dir,
    SolverType solver = SolverType::GPU  // Add solver type parameter with GPU default
) {
    const int N = positions.size();
    for (int i = 0; i < num_freqs; ++i) {
        double freq = (num_freqs == 1) ? f_start : f_start + i * (f_end - f_start) / (num_freqs - 1);
        
        // Create output filename in the specific output directory
        std::ostringstream filename;
        filename << output_dir << "/output_freq_" << std::scientific << std::setprecision(2) << freq << ".csv";
        
        struct stat buffer;
        if (stat(filename.str().c_str(), &buffer) == 0) {
            std::cout << "Skipping frequency " << freq << " Hz - output file " << filename.str() << " already exists\n";
            continue;
        } else if (errno != ENOENT) {
            // If error is not "file doesn't exist", we have a problem
            std::cerr << "Error checking file " << filename.str() << ": " << strerror(errno) << std::endl;
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

            // Convert incident field to cuDoubleComplex for GPU/CPU solvers
            std::vector<cuDoubleComplex> b_cuda(2 * N);
            for (int i = 0; i < 2 * N; ++i) {
                b_cuda[i] = make_cuDoubleComplex(std::real(inc_field[i]), std::imag(inc_field[i]));
            }

            // For GPU solver, transpose matrix (GPU expects column-major format)
            // For CPU solver, keep in row-major format
            if (solver == SolverType::GPU) {
                std::vector<cuDoubleComplex> A_transposed(2 * N * 2 * N);
                for (int i = 0; i < 2 * N; ++i) {
                    for (int j = 0; j < 2 * N; ++j) {
                        A_transposed[i * 2 * N + j] = make_cuDoubleComplex(
                            std::real(A_host[j * 2 * N + i]),
                            std::imag(A_host[j * 2 * N + i])
                        );
                    }
                }
                cudaMemcpy(A_device, A_transposed.data(), sizeof(cuDoubleComplex) * 2 * N * 2 * N, cudaMemcpyHostToDevice);
            } else {
                // For CPU solver, just convert to cuDoubleComplex format
                std::vector<cuDoubleComplex> A_cpu(2 * N * 2 * N);
                for (int i = 0; i < 2 * N * 2 * N; ++i) {
                    A_cpu[i] = make_cuDoubleComplex(std::real(A_host[i]), std::imag(A_host[i]));
                }
                solve_cpu(A_cpu.data(), b_cuda.data(), 2 * N);
            }

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

// ---- Create directory if it doesn't exist ----
bool create_directory(const std::string& path) {
    // Check if directory exists
    DIR* dir = opendir(path.c_str());
    if (dir) {
        closedir(dir);
        return true;
    }
    
    // Create directory with read/write/execute permissions for owner
    int status = mkdir(path.c_str(), S_IRWXU);
    if (status != 0) {
        std::cerr << "Error creating directory " << path << ": " << strerror(errno) << std::endl;
        return false;
    }
    return true;
}

// ---- Get filename without extension ----
std::string get_filename_without_ext(const std::string& filepath) {
    size_t lastSlash = filepath.find_last_of("/\\");
    
    // Extract just the filename part (after the last slash if it exists)
    std::string filename = (lastSlash == std::string::npos) ? filepath : filepath.substr(lastSlash + 1);
    
    // Remove extension if it exists
    size_t lastDot = filename.find_last_of(".");
    if (lastDot != std::string::npos) {
        filename = filename.substr(0, lastDot);
    }
    
    return filename;
}

// ---- Process Single CSV File ----
void process_csv_file(const std::string& csv_path, SolverType solver = SolverType::GPU) {
    std::vector<vec3> positions;
    std::vector<LorentzianParams> params_00, params_05, params_50, params_55;
    std::vector<double> angles;

    // Create output directory with same name as CSV (without extension)
    std::string output_dir = get_filename_without_ext(csv_path);
    if (!create_directory(output_dir)) {
        return;
    }

    // Load the dipole parameters from CSV
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file " << csv_path << std::endl;
        return;
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
        // std::cout << values[0] << "\n";

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
        std::cerr << "Error: No dipoles loaded from input file " << csv_path << std::endl;
        return;
    }

    std::cout << "Processing " << csv_path << " with " << N << " dipoles" << std::endl;
    
    // Run the simulation with the loaded parameters
    auto simulation_start = std::chrono::high_resolution_clock::now();
    run_simulation(150e12, 350e12, 50, positions, params_00, params_05, params_50, params_55, angles, output_dir, solver);
    auto simulation_end = std::chrono::high_resolution_clock::now();
    auto simulation_duration = std::chrono::duration_cast<std::chrono::seconds>(simulation_end - simulation_start);
    std::cout << "Finished processing " << csv_path << " in " << simulation_duration.count() << " seconds" << std::endl;
}

// ---- Main Function ----
int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> [solver_type]\n";
        std::cerr << "  input_directory: Path to directory containing CSV files\n";
        std::cerr << "  solver_type: 'cpu' or 'gpu' (default: gpu)\n";
        std::cerr << "  CSV format: x,y,z,f0_00,gamma_00,A_00,...,angle\n";
        return 1;
    }

    // Determine solver type
    SolverType solver = SolverType::GPU;  // Default to GPU
    if (argc == 3) {
        std::string solver_arg = argv[2];
        if (solver_arg == "cpu") {
            solver = SolverType::CPU;
        } else if (solver_arg == "gpu") {
            solver = SolverType::GPU;
        } else {
            std::cerr << "Invalid solver type. Must be 'cpu' or 'gpu'\n";
            return 1;
        }
    }

    // Open directory
    DIR* dir = opendir(argv[1]);
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory " << argv[1] << ": " << strerror(errno) << std::endl;
        return 1;
    }

    // Update process_csv_file to use selected solver
    auto process_file = [solver](const std::string& filepath) {
        process_csv_file(filepath, solver);
    };

    // Read directory entries
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        
        // Skip . and .. directories
        if (filename == "." || filename == "..") {
            continue;
        }
        
        // Check if file ends with .csv
        if (filename.length() >= 4 && 
            filename.compare(filename.length() - 4, 4, ".csv") == 0) {
            
            std::string filepath = std::string(argv[1]) + "/" + filename;
            process_file(filepath);
        }
    }

    closedir(dir);
    return 0;
}
