#include "lorentzian.hpp"
#include <complex>

const std::complex<double> I(0.0, 1.0);

// Constants for Lorentzian parameters
const LorentzianParams LORENTZ_00 = {EEXX_A_PARAM, EEXX_B_PARAM, EEXX_C_PARAM, EEXX_F0, EEXX_GAMMA_PARAM, LorentzianForm::STANDARD};
const LorentzianParams LORENTZ_05 = {EMXZ_A_PARAM, EMXZ_B_PARAM, EMXZ_C_PARAM, EMXZ_F0, EMXZ_GAMMA_PARAM, LorentzianForm::IF_NUMERATOR};
const LorentzianParams LORENTZ_50 = {MEZX_A_PARAM, MEZX_B_PARAM, MEZX_C_PARAM, MEZX_F0, MEZX_GAMMA_PARAM, LorentzianForm::NEG_IF_NUMERATOR};
const LorentzianParams LORENTZ_55 = {MMZZ_A_PARAM, MMZZ_B_PARAM, MMZZ_C_PARAM, MMZZ_F0, MMZZ_GAMMA_PARAM, LorentzianForm::F2_NUMERATOR};

std::complex<double> lorentz_alpha_params(double f, const LorentzianParams& params) {
    std::complex<double> denom = (params.F0 * params.F0 - f * f) - I * f * params.gamma;
    std::complex<double> numerator;
    std::complex<double> scale_factor;
    
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
            scale_factor = 1.0 / Z_0;
            break;
        case LorentzianForm::F2_NUMERATOR:
            numerator = f * f * params.A;
            scale_factor = 1.0;
            break;
    }
    
    std::complex<double> norm_alpha = numerator / denom + params.B + params.C * f;
    return scale_factor * norm_alpha;
}

void generate_polarizability_matrix(std::complex<double> alpha[2][2], double freq, double f0_offset) {
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
    
    // Set the four elements of the 2x2 matrix using the Lorentzian corners
    alpha[0][0] = lorentz_alpha_params(freq, params_00);  // EE
    alpha[0][1] = lorentz_alpha_params(freq, params_05);  // EM
    alpha[1][0] = lorentz_alpha_params(freq, params_50);  // HE
    alpha[1][1] = lorentz_alpha_params(freq, params_55);  // HM
}