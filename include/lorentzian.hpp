#ifndef LORENTZIAN_HPP
#define LORENTZIAN_HPP

#include <complex>
#include "constants.hpp"

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

// Function declarations
std::complex<double> lorentz_alpha_params(double f, const LorentzianParams& params);
void generate_polarizability_matrix(std::complex<double> alpha[2][2], double freq, double f0_offset);

// Constants for Lorentzian parameters
extern const LorentzianParams LORENTZ_00;
extern const LorentzianParams LORENTZ_05;
extern const LorentzianParams LORENTZ_50;
extern const LorentzianParams LORENTZ_55;

#endif // LORENTZIAN_HPP