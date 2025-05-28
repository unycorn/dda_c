#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>
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

using mat6x6 = std::complex<double>[6][6];
constexpr std::complex<double> I(0.0, 1.0);

// Function declarations
void generate_polarizability_matrix(mat6x6& alpha, double freq, double f0_offset);
void rotate_polarizability_matrix(mat6x6& alpha, double theta);
void create_rotation_matrix(std::complex<double> out[6][6], double theta);
void matrix_multiply(std::complex<double> result[6][6], 
                    const std::complex<double> a[6][6], 
                    const std::complex<double> b[6][6]);

void run_simulation(
    double f_start,
    double f_end,
    int num_freqs,
    const std::vector<vec3>& positions,
    int N,
    double spacing,
    double disorder,
    double f0_disorder,
    double angle_disorder,
    unsigned int seed);

// Add these after the existing includes, before function declarations
enum class LorentzianForm {
    STANDARD,       // top left: A/(f0² - f² - if𝛾)
    IF_NUMERATOR,   // top right: ifA/(f0² - f² - if𝛾)
    NEG_IF_NUMERATOR, // bottom left: -ifA/(f0² - f² - if𝛾)
    F2_NUMERATOR    // bottom right: f²A/(f0² - f² - if𝛾)
};

struct LorentzianParams {
    double A;
    double B;
    double C;
    double F0;
    double gamma;
    LorentzianForm form;
};

// Four sets of Lorentzian parameters for the corner elements
const LorentzianParams LORENTZ_00 = {EEXX_A_PARAM, EEXX_B_PARAM, EEXX_C_PARAM, EEXX_F0, EEXX_GAMMA_PARAM, LorentzianForm::STANDARD};
const LorentzianParams LORENTZ_05 = {EMXZ_A_PARAM, EMXZ_B_PARAM, EMXZ_C_PARAM, EMXZ_F0, EMXZ_GAMMA_PARAM, LorentzianForm::IF_NUMERATOR};
const LorentzianParams LORENTZ_50 = {MEZX_A_PARAM, MEZX_B_PARAM, MEZX_C_PARAM, MEZX_F0, MEZX_GAMMA_PARAM, LorentzianForm::NEG_IF_NUMERATOR};
const LorentzianParams LORENTZ_55 = {MMZZ_A_PARAM, MMZZ_B_PARAM, MMZZ_C_PARAM, MMZZ_F0, MMZZ_GAMMA_PARAM, LorentzianForm::F2_NUMERATOR};

// Function to compute Lorentzian polarizability with arbitrary parameters
std::complex<double> lorentz_alpha_params(double f, const LorentzianParams& params) {
    std::complex<double> denom = (params.F0 * params.F0 - f * f) - I * f * params.gamma;
    std::complex<double> numerator;
    std::complex<double> scale_factor;
    
    // Handle different forms of the numerator
    switch (params.form) {
        case LorentzianForm::STANDARD:
            numerator = params.A;
            scale_factor = EPSILON_0;
            break;
        case LorentzianForm::IF_NUMERATOR:
            numerator = I * f * params.A;
            scale_factor = EPSILON_0 * Z_0;
            break;
        case LorentzianForm::NEG_IF_NUMERATOR:
            numerator = -I * f * params.A;
            scale_factor = EPSILON_0;
            break;
        case LorentzianForm::F2_NUMERATOR:
            numerator = f * f * params.A;
            scale_factor = EPSILON_0;
            break;
    }
    
    std::complex<double> norm_alpha = numerator / denom + params.B + params.C * f;
    return scale_factor * norm_alpha;
}

// Function to create a rotation matrix for the 6x6 polarizability tensor
void create_rotation_matrix(std::complex<double> out[6][6], double theta) {
    // Typically, rotation matrices are 3x3, but here we need a 6x6 matrix
    // which rotates all four tensors in the 6x6 "supertensor"
    // this matrix is like a tensor product of the 3x3 rotation matrix with the identity matrix
    //         R6x6 = R(3x3) ⊗ I_2 =  | R(3x3)  0      |
    //                                | 0       R(3x3) |

    // Clear the matrix
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            out[i][j] = 0.0;
        }
    }
    
    // Fill in rotation matrix elements for electric part (top-left 3x3)
    out[0][0] = std::cos(theta);
    out[0][1] = -std::sin(theta);
    out[1][0] = std::sin(theta);
    out[1][1] = std::cos(theta);
    out[2][2] = 1.0;  // z-component unchanged

    // Fill in rotation matrix elements for magnetic part (bottom-right 3x3)
    out[3][3] = std::cos(theta);
    out[3][4] = -std::sin(theta);
    out[4][3] = std::sin(theta);
    out[4][4] = std::cos(theta);
    out[5][5] = 1.0;
}

// Function to multiply 6x6 complex matrices: result = a * b (or a @ b in Python notation)
void matrix_multiply(std::complex<double> result[6][6], 
                    const std::complex<double> a[6][6], 
                    const std::complex<double> b[6][6]) {
    std::complex<double> temp[6][6];
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            temp[i][j] = 0;
            for (int k = 0; k < 6; ++k) {
                temp[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            result[i][j] = temp[i][j];
        }
    }
}

// Fills a 2D grid of dipoles in the xy-plane (z = 0)
void generate_positions(vec3* positions, int N_width, int N_height, double spacing) {
    int i = 0;
    for (int i_x = 0; i_x < N_width; ++i_x) {
        for (int i_y = 0; i_y < N_height; ++i_y) {
            positions[i++] = vec3{
                i_x * spacing,
                i_y * spacing,
                0.0
            };
        }
    }
}

// Generates a disordered dipole grid in the xy-plane with optional RNG seed
void generate_disordered_positions(vec3* positions, int N_width, int N_height, double spacing, double rms_displacement, unsigned int seed = 0) {
    std::default_random_engine rng(seed);  // seed controls reproducibility
    std::normal_distribution<double> normal(0.0, rms_displacement);  // standard deviation = RMS displacement

    int i = 0;
    for (int i_x = 0; i_x < N_width; ++i_x) {
        for (int i_y = 0; i_y < N_height; ++i_y) {
            positions[i] = vec3{
                i_x * spacing + normal(rng),
                i_y * spacing + normal(rng),
                0.0
            };
            i += 1;
        }
    }
}

// Disordered Lorentzian polarizability function with pre-generated F0 value
// std::complex<double> disordered_lorentz_alpha(double f, double disordered_f0) {
//     std::complex<double> denom = (disordered_f0 * disordered_f0 - f * f) - I * f * GAMMA_PARAM;
//     std::complex<double> norm_alpha = A_PARAM / denom + B_PARAM + C_PARAM * f;
//     return norm_alpha * EPSILON_0;
// }

void run_simulation(
    double f_start,
    double f_end,
    int num_freqs,
    const std::vector<vec3>& positions,
    int N,
    double spacing,
    double disorder,
    double f0_disorder,
    double angle_disorder,
    unsigned int seed
) {
    // Create RNG once for the whole simulation
    std::default_random_engine rng(seed);

    // Generate disordered F0 values for each dipole once
    std::vector<double> f0_offsets(N);
    std::normal_distribution<double> normal_f0(0.0, f0_disorder);
    for (int j = 0; j < N; ++j) {
        f0_offsets[j] = normal_f0(rng);
    }
    
    // Generate random rotation angles for each dipole once
    std::vector<double> rotation_angles(N);
    std::normal_distribution<double> normal_angle(0.0, angle_disorder);
    for (int j = 0; j < N; ++j) {
        rotation_angles[j] = normal_angle(rng);
    }

    for (int i = 0; i < num_freqs; ++i) {
        // Replace the existing line with this:
        double freq = (num_freqs == 1) ? f_start : f_start + i * (f_end - f_start) / (num_freqs - 1);
        
        // Check if output file already exists
        std::ostringstream filename;
        filename << "output/output_(" << std::scientific << std::setprecision(2)
                 << freq << ")_(" << disorder * 1e9 << "nm)_(" << f0_disorder << "Hz)_(" 
                 << angle_disorder << "rad)_seed" << seed << ".csv";
        
        if (std::ifstream(filename.str()).good()) {
            std::cout << "Skipping frequency " << freq << " Hz - output file already exists\n";
            continue;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        double wavelength = C_LIGHT / freq;
        double k = 2.0 * M_PI / wavelength;

        std::vector<mat6x6> alpha(N);
        
        for (int j = 0; j < N; ++j) {
            // Generate the base polarizability matrix
            generate_polarizability_matrix(alpha[j], freq, f0_offsets[j]);
            
            // Rotate the polarizability matrix
            rotate_polarizability_matrix(alpha[j], rotation_angles[j]);
        }

        std::vector<std::complex<double>> A_host(6 * N * 6 * N, std::complex<double>(0.0, 0.0));
        cuDoubleComplex* A_device = get_full_interaction_matrix(A_host.data(), positions.data(), alpha.data(), N, k);
        std::cout << "freq " << freq << ": Finished Computing Interaction Matrix!\n";

        try {
            // Initialize incident field (now 6N components, but only E-field is non-zero)
            std::vector<std::complex<double>> inc_field(6 * N, std::complex<double>(0.0, 0.0));
            for (int j = 0; j < N; ++j) {
                double phase = k * positions[j].z;
                auto val = std::exp(I * phase);
                inc_field[6 * j + 0] = val;  // x-component of E-field
                inc_field[6 * j + 4] = val / Z_0;  // y-component of H-field
            }

            std::vector<std::complex<double>> b(6 * N);
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < 6; ++j) {
                    b[i * 6 + j] = inc_field[i * 6 + j];
                }
            }

            std::vector<cuDoubleComplex> b_cuda(6 * N);
            for (int i = 0; i < 6 * N; ++i) {
                b_cuda[i] = make_cuDoubleComplex(std::real(b[i]), std::imag(b[i]));
            }

            solve_gpu(A_device, b_cuda.data(), 6 * N);

            for (int i = 0; i < 6 * N; ++i) {
                b[i] = std::complex<double>(cuCreal(b_cuda[i]), cuCimag(b_cuda[i]));
            }

            // Clean up GPU resources before next iteration
            if (A_device != nullptr) {
                cudaFree(A_device);
                A_device = nullptr;
            }

            // For output, we only use the electric part of the response
            write_polarizations(filename.str().c_str(), b.data(), positions, alpha, N);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto ms_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            std::cout << "Elapsed: " << ms_duration.count() << " ms\n";

        } catch (const std::exception& e) {
            std::cerr << "Error during frequency " << freq << ": " << e.what() << std::endl;
            // Make sure to clean up even if there's an error
            if (A_device != nullptr) {
                cudaFree(A_device);
                A_device = nullptr;
            }
            throw;
        }
    }
}

// Function to generate the base polarizability matrix before rotation
void generate_polarizability_matrix(mat6x6& alpha, double freq, double f0_offset) {
    // Initialize matrix with specific values
    alpha[0][0] = std::complex<double>(8.07055829e-33, -9.79503385e-33);
    alpha[0][1] = std::complex<double>(3.65970195e-34, 1.79213602e-34);
    alpha[0][2] = std::complex<double>(-3.42179166e-34, -1.43045868e-34);
    alpha[0][3] = std::complex<double>(1.28814924e-31, 5.98763814e-32);
    alpha[0][4] = std::complex<double>(-1.09555273e-31, -7.52727972e-32);
    alpha[0][5] = std::complex<double>(-2.90720115e-31, 4.12464649e-31);

    alpha[1][0] = std::complex<double>(6.86773642e-36, 4.62497388e-35);
    alpha[1][1] = std::complex<double>(9.18647046e-33, -1.58077714e-32);
    alpha[1][2] = std::complex<double>(4.79136977e-35, -6.93842326e-35);
    alpha[1][3] = std::complex<double>(-3.60016703e-32, 3.35733168e-32);
    alpha[1][4] = std::complex<double>(6.27119725e-33, -9.06211235e-33);
    alpha[1][5] = std::complex<double>(6.58912022e-33, 3.77031418e-34);

    alpha[2][0] = std::complex<double>(3.20518649e-35, 4.83199981e-35);
    alpha[2][1] = std::complex<double>(-6.63902407e-36, -7.63630472e-35);
    alpha[2][2] = std::complex<double>(4.78607812e-33, -1.24023760e-33);
    alpha[2][3] = std::complex<double>(-9.27687727e-33, -1.15924308e-32);
    alpha[2][4] = std::complex<double>(-1.32136374e-32, -1.32012163e-32);
    alpha[2][5] = std::complex<double>(1.30369698e-32, 1.06981617e-32);

    alpha[3][0] = std::complex<double>(-7.23699005e-26, 2.92959883e-26);
    alpha[3][1] = std::complex<double>(9.84648698e-26, -4.19393866e-26);
    alpha[3][2] = std::complex<double>(7.61824424e-26, -2.23788470e-26);
    alpha[3][3] = std::complex<double>(1.42280228e-23, -1.77766145e-23);
    alpha[3][4] = std::complex<double>(2.71641322e-23, -1.11825183e-23);
    alpha[3][5] = std::complex<double>(-2.72337292e-23, 1.12519748e-23);

    alpha[4][0] = std::complex<double>(-7.45532094e-26, 2.82179646e-26);
    alpha[4][1] = std::complex<double>(6.01649197e-26, -2.55983084e-26);
    alpha[4][2] = std::complex<double>(-5.73631923e-26, 2.59831922e-26);
    alpha[4][3] = std::complex<double>(2.20353103e-23, -9.66804168e-24);
    alpha[4][4] = std::complex<double>(1.04816403e-23, -1.36726372e-23);
    alpha[4][5] = std::complex<double>(-2.03190690e-23, 9.22192600e-24);

    alpha[5][0] = std::complex<double>(1.65819462e-25, -3.28110024e-25);
    alpha[5][1] = std::complex<double>(-7.19345233e-26, -1.64425006e-27);
    alpha[5][2] = std::complex<double>(6.58868710e-26, 6.95051960e-27);
    alpha[5][3] = std::complex<double>(-2.53065675e-23, -2.13721408e-24);
    alpha[5][4] = std::complex<double>(2.71420434e-23, 2.59368101e-24);
    alpha[5][5] = std::complex<double>(-7.67688480e-23, -1.10355170e-22);
    
    // Create temporary params with the offset added to F0
    LorentzianParams params_00 = LORENTZ_00;
    LorentzianParams params_05 = LORENTZ_05;
    LorentzianParams params_50 = LORENTZ_50;
    LorentzianParams params_55 = LORENTZ_55;
    
    // Add the same offset to all resonances
    params_00.F0 += f0_offset;
    params_05.F0 += f0_offset;
    params_50.F0 += f0_offset;
    params_55.F0 += f0_offset;
    
    // Set the four corner Lorentzian elements with offset frequencies
    alpha[0][0] = lorentz_alpha_params(freq, params_00);
    alpha[0][5] = lorentz_alpha_params(freq, params_05);
    alpha[5][0] = lorentz_alpha_params(freq, params_50);
    alpha[5][5] = lorentz_alpha_params(freq, params_55);
}

// Function to handle rotation of polarizability matrix
void rotate_polarizability_matrix(mat6x6& alpha, double theta) {
    std::complex<double> rotation[6][6];
    std::complex<double> rotation_T[6][6];
    std::complex<double> temp[6][6];
    
    create_rotation_matrix(rotation, theta);
    create_rotation_matrix(rotation_T, -theta);
    
    matrix_multiply(temp, rotation, alpha);
    matrix_multiply(alpha, temp, rotation_T);
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <disorder_nm> <f0_disorder_Hz> <angle_disorder_rad> <seed>\n";
        std::cerr << "  disorder_nm: RMS displacement in nanometers\n";
        std::cerr << "  f0_disorder_Hz: RMS disorder in F0 frequency (Hz)\n";
        std::cerr << "  angle_disorder_rad: RMS disorder in rotation angle (radians)\n";
        std::cerr << "  seed: Random number generator seed\n";
        return 1;
    }

    double disorder = std::stod(argv[1]) * 1e-9;
    double f0_disorder = std::stod(argv[2]);
    double angle_disorder = std::stod(argv[3]);
    unsigned int seed = static_cast<unsigned int>(std::stoul(argv[4]));

    const int N_width = 2;
    const int N_height = 1;
    const int N = N_width * N_height;
    const double spacing = 300e-9;

    std::vector<vec3> positions(N);
    generate_disordered_positions(positions.data(), N_width, N_height, spacing, disorder, seed);

    run_simulation(150e12, 350e12, 1, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);
    // run_simulation(150e12, 350e12, 80, positions, N, spacing, disorder, f0_disorder, angle_disorder, seed);

    return 0;
}
