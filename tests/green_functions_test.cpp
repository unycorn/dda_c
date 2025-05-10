#include <gtest/gtest.h>
#include <complex>
#include "../include/interaction.hpp"
#include "../include/vector3.hpp"
#include "../include/constants.hpp"

// Test fixture for Green's function tests
class GreenFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for all tests
        k = 2.0 * M_PI / 500e-9;  // wavelength = 500nm
        r1 = {0.0, 0.0, 0.0};
        r2 = {300e-9, 0.0, 0.0};  // 300nm separation along x-axis
    }

    double k;
    vec3 r1, r2;
    const double tolerance = 1e-15;  // Tolerance for floating point comparisons
};

// Test E-E Green's function for known configuration
TEST_F(GreenFunctionTest, GreenEEDipoleTest) {
    std::complex<double> G_EE[3][3];
    green_E_E_dipole(G_EE, r2, r1, k);
    
    // For dipoles separated along x-axis, we expect:
    // - G_xx should be real and negative
    // - G_yy and G_zz should be equal
    // - Off-diagonal terms should be zero
    EXPECT_NEAR(std::imag(G_EE[0][0]), 0.0, tolerance);
    EXPECT_LT(std::real(G_EE[0][0]), 0.0);
    EXPECT_NEAR(std::abs(G_EE[0][1]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[0][2]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[1][0]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[2][0]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[1][2]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[2][1]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_EE[1][1] - G_EE[2][2]), 0.0, tolerance);
}

// Test H-E Green's function for known configuration
TEST_F(GreenFunctionTest, GreenHEDipoleTest) {
    std::complex<double> G_HE[3][3];
    green_H_E_dipole(G_HE, r2, r1, k);
    
    // For dipoles separated along x-axis:
    // - Only G_yz and G_zy should be non-zero
    // - They should be equal in magnitude but opposite in sign
    EXPECT_NEAR(std::abs(G_HE[0][0]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_HE[0][1]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_HE[0][2]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_HE[1][0]), 0.0, tolerance);
    EXPECT_NEAR(std::abs(G_HE[2][0]), 0.0, tolerance);
    EXPECT_GT(std::abs(G_HE[1][2]), 0.0);
    EXPECT_GT(std::abs(G_HE[2][1]), 0.0);
    EXPECT_NEAR(G_HE[1][2], -G_HE[2][1], tolerance);
}

// Test full 6x6 interaction matrix
TEST_F(GreenFunctionTest, FullInteractionMatrixTest) {
    const int N = 2;  // Two dipoles
    std::vector<vec3> positions = {r1, r2};
    
    // Create a simple diagonal polarizability matrix
    std::complex<double> alpha[N][6][6];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int m = 0; m < 6; ++m) {
                alpha[i][j][m] = (j == m) ? std::complex<double>(1e-35, 0.0) : std::complex<double>(0.0, 0.0);
            }
        }
    }
    
    // Build full interaction matrix
    std::vector<std::complex<double>> A(6*N * 6*N);
    get_full_interaction_matrix(A.data(), positions.data(), alpha, N, k);
    
    // Test properties of the interaction matrix:
    // 1. The diagonal blocks should contain the inverse polarizability
    // 2. The off-diagonal blocks should contain the Green's function tensors
    // 3. The matrix should be symmetric (reciprocity)
    
    // Test diagonal block structure
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(std::abs(A[i*6*N + i] - 1e35), 0.0, 1e20);
    }
    
    // Test symmetry between dipoles
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            std::complex<double> G12 = A[i*6*N + (j + 6)];
            std::complex<double> G21 = A[(i + 6)*6*N + j];
            EXPECT_NEAR(std::abs(G12 - G21), 0.0, tolerance);
        }
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}