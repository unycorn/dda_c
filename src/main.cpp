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
    const std::string& output_dir,  // Add output directory parameter
    bool use_gaussian_beam,         // Flag to enable Gaussian beam
    double w0                        // Beam waist parameter for Gaussian beam
) {
    const int N = positions.size();
    
    // Open CSV file for power logging if using Gaussian beam
    std::ofstream power_csv;
    if (use_gaussian_beam) {
        std::string power_csv_path = output_dir + "/power_data.csv";
        power_csv.open(power_csv_path);
        if (!power_csv.is_open()) {
            std::cerr << "Warning: Could not open power CSV file: " << power_csv_path << std::endl;
        } else {
            // Write header
            power_csv << "frequency_Hz,incident_power,absorbed_power,extinguished_power\n";
            power_csv << std::scientific << std::setprecision(10);
        }
    }
    
    for (int i = 0; i < num_freqs; ++i) {
        double freq = (num_freqs == 1) ? f_start : f_start + i * (f_end - f_start) / (num_freqs - 1);
        
        // Create output filename in the specific output directory
        std::ostringstream filename;
        filename << output_dir << "/output_freq_" << std::scientific << std::setprecision(5) << freq << ".pols";
        std::ostringstream csvfilename;
        csvfilename << output_dir << "/output_freq_" << std::scientific << std::setprecision(5) << freq << ".csv";
        
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
        // cuDoubleComplex* A_device = get_full_interaction_matrix_scalar_1Dperiodic(A_host.data(), positions.data(), alpha.data(), angles.data(), N, k, vec3{1365.0e-9, 0.0, 0.0}, 1000);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        try {
            // Initialize incident field (2N components - one electric and one magnetic scalar per dipole)
            std::vector<std::complex<double>> inc_field(2 * N, std::complex<double>(0.0, 0.0));
            
            if (use_gaussian_beam) {
                // Gaussian beam profile
                std::cout << "Using Gaussian beam with w0 = " << w0 << " m\n";
                for (int j = 0; j < N; ++j) {
                    double rho2 = positions[j].x * positions[j].x + positions[j].y * positions[j].y;
                    inc_field[2 * j + 0] = cos(angles[j]) * std::exp(-rho2 / (2 * w0 * w0)) * std::complex<double>(1.0, 0.0); // Ex
                    inc_field[2 * j + 1] = std::complex<double>(0.0, 0.0); // Hz
                }
            } else {
                // Default plane wave
                for (int j = 0; j < N; ++j) {
                    inc_field[2 * j + 0] = cos(angles[j]) * std::complex<double>(1.0, 0.0); // Ex
                    inc_field[2 * j + 1] = std::complex<double>(0.0, 0.0); // Hz
                }
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

            // Compute absorbed power: omega/2 * imag(sum over all dipoles of (p_j*, m_j*) * alpha_j^(-1) * (p_j; m_j))
            // Compute extinguished power: omega/2 * imag(sum over all dipoles of (E_j*, H_j*) * (p_j; m_j))
            double omega = 2.0 * M_PI * freq;
            std::complex<double> absorbed_power_sum(0.0, 0.0);
            std::complex<double> extinguished_power_sum(0.0, 0.0);
            
            // Compute absorbed and extinguished power
            for (int j = 0; j < N; ++j) {
                std::complex<double> px = b[2 * j + 0];     // Electric polarization for dipole j
                std::complex<double> mz = b[2 * j + 1];     // Magnetic polarization for dipole j
                
                // Get incident field components for dipole j
                std::complex<double> Ex_inc = inc_field[2 * j + 0];
                std::complex<double> Hz_inc = inc_field[2 * j + 1];
                
                // Compute extinguished power contribution: (E_j*, H_j*) * (p_j; m_j)
                std::complex<double> extinguished_power_complex = std::conj(Ex_inc) * px + MU_0 * std::conj(Hz_inc) * mz;
                extinguished_power_sum += extinguished_power_complex;
                
                // Invert 2x2 alpha matrix for dipole j using determinant method
                std::complex<double> det = alpha[j][0][0] * alpha[j][1][1] - alpha[j][0][1] * alpha[j][1][0];
                std::complex<double> alpha_inv[2][2];
                alpha_inv[0][0] = alpha[j][1][1] / det;
                alpha_inv[0][1] = -alpha[j][0][1] / det;
                alpha_inv[1][0] = -alpha[j][1][0] / det;
                alpha_inv[1][1] = alpha[j][0][0] / det;
                
                // Take Hermitian conjugate (dagger) of alpha_inv
                std::complex<double> alpha_inv_dagger[2][2];
                alpha_inv_dagger[0][0] = std::conj(alpha_inv[0][0]);
                alpha_inv_dagger[0][1] = std::conj(alpha_inv[1][0]);
                alpha_inv_dagger[1][0] = std::conj(alpha_inv[0][1]);
                alpha_inv_dagger[1][1] = std::conj(alpha_inv[1][1]);
                
                // Calculate (px*, mz*) * alpha_inv^† * (px; mz) for dipole j
                std::complex<double> Ex_star = std::conj(px) * alpha_inv_dagger[0][0] + std::conj(mz) * alpha_inv_dagger[1][0];
                std::complex<double> Hz_star = std::conj(px) * alpha_inv_dagger[0][1] + std::conj(mz) * alpha_inv_dagger[1][1];
                std::complex<double> absorbed_power_complex = Ex_star * px + MU_0 * Hz_star * mz;
                
                absorbed_power_sum += absorbed_power_complex;
            }
            
            double absorbed_power_total = (omega / 2.0) * std::imag(absorbed_power_sum);
            double extinguished_power_total = (omega / 2.0) * std::imag(extinguished_power_sum);
            
            // Compute incident power for Gaussian beam: P_inc = pi * w0^2 / (2 * Z_0)
            double incident_power = 0.0;
            if (use_gaussian_beam) {
                incident_power = M_PI * w0 * w0 / (2.0 * Z_0);
            }
            
            std::cout << "Absorbed power at " << freq << " Hz: " << absorbed_power_total << std::endl;
            std::cout << "Extinguished power at " << freq << " Hz: " << extinguished_power_total << std::endl;
            if (use_gaussian_beam) {
                std::cout << "Incident power (Gaussian beam): " << incident_power << std::endl;
            }

            // Write to power CSV if using Gaussian beam
            if (use_gaussian_beam && power_csv.is_open()) {
                power_csv << freq << "," << incident_power << "," << absorbed_power_total << "," << extinguished_power_total << "\n";
            }

            if (A_device != nullptr) {
                cudaFree(A_device);
                A_device = nullptr;
            }

            // write_polarizations(csvfilename.str().c_str(), b.data(), positions, alpha, N);  // Original plaintext writer
            write_polarizations_binary(filename.str().c_str(), b.data(), positions, alpha, N, freq, absorbed_power_total);  // New binary writer with frequency and absorption

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
    
    // Close power CSV file if it was opened
    if (power_csv.is_open()) {
        power_csv.close();
        std::cout << "Power data saved to " << output_dir << "/power_data.csv\n";
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
void process_csv_file(const std::string& csv_path, double f_start, double f_end, int num_freqs, 
                     bool use_gaussian_beam, double w0) {
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
    run_simulation(f_start, f_end, num_freqs, positions, params_00, params_05, params_50, params_55, angles, output_dir, use_gaussian_beam, w0);
    auto simulation_end = std::chrono::high_resolution_clock::now();
    auto simulation_duration = std::chrono::duration_cast<std::chrono::seconds>(simulation_end - simulation_start);
    std::cout << "Finished processing " << csv_path << " in " << simulation_duration.count() << " seconds" << std::endl;
}

// ---- Main Function ----
int main(int argc, char* argv[]) {
    if (argc != 5 && argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <f_start> <f_end> <num_freqs> [--gaussian-beam <w0>]\n";
        std::cerr << "  input_directory: Path to directory containing CSV files\n";
        std::cerr << "  f_start: Starting frequency in Hz\n";
        std::cerr << "  f_end: Ending frequency in Hz\n";
        std::cerr << "  num_freqs: Number of frequency points\n";
        std::cerr << "  --gaussian-beam: Optional flag to use Gaussian beam instead of plane wave\n";
        std::cerr << "  w0: Beam waist parameter in meters (required if --gaussian-beam is used)\n";
        std::cerr << "  CSV format: x,y,z,f0_00,gamma_00,A_00,...,angle\n";
        return 1;
    }

    // Parse frequency parameters
    double f_start = std::stod(argv[2]);
    double f_end = std::stod(argv[3]);
    int num_freqs = std::stoi(argv[4]);
    
    // Parse optional Gaussian beam parameters
    bool use_gaussian_beam = false;
    double w0 = 0.0;
    
    if (argc == 7) {
        std::string flag = argv[5];
        if (flag == "--gaussian-beam") {
            use_gaussian_beam = true;
            w0 = std::stod(argv[6]);
            if (w0 <= 0) {
                std::cerr << "Error: Beam waist w0 must be positive.\n";
                return 1;
            }
            std::cout << "Using Gaussian beam mode with w0 = " << w0 << " m\n";
        } else {
            std::cerr << "Error: Unknown flag '" << flag << "'. Use --gaussian-beam.\n";
            return 1;
        }
    } else {
        std::cout << "Using plane wave mode (default)\n";
    }

    // Validate frequency parameters
    if (f_start <= 0 || f_end <= 0 || num_freqs <= 0 || f_start >= f_end) {
        std::cerr << "Error: Invalid frequency parameters.\n";
        std::cerr << "f_start and f_end must be positive, f_start must be less than f_end,\n";
        std::cerr << "and num_freqs must be positive.\n";
        return 1;
    }

    // Open directory
    DIR* dir = opendir(argv[1]);
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory " << argv[1] << ": " << strerror(errno) << std::endl;
        return 1;
    }

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
            process_csv_file(filepath, f_start, f_end, num_freqs, use_gaussian_beam, w0);
        }
    }

    closedir(dir);
    return 0;
}