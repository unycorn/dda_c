// dipole_field.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include "dipole_field.hpp"

__device__ vec3 device_vec3_sub(vec3 a, vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ double device_vec3_norm(vec3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ vec3 device_vec3_normalize(vec3 v) {
    double norm = device_vec3_norm(v);
    return {v.x / norm, v.y / norm, v.z / norm};
}

__device__ cuDoubleComplex dot_cvec3(vec3 a, cvec3 b) {
    return cuCadd(cuCadd(
        cuCmul(make_cuDoubleComplex(a.x, 0.0), b.x),
        cuCmul(make_cuDoubleComplex(a.y, 0.0), b.y)),
        cuCmul(make_cuDoubleComplex(a.z, 0.0), b.z));
}

__device__ cvec3 cross_cvec3(cvec3 a, cvec3 b) {
    return {
        cuCsub(cuCmul(a.y, b.z), cuCmul(a.z, b.y)),
        cuCsub(cuCmul(a.z, b.x), cuCmul(a.x, b.z)),
        cuCsub(cuCmul(a.x, b.y), cuCmul(a.y, b.x))
    };
}

__device__ cvec3 scale_cvec3(cvec3 v, cuDoubleComplex s) {
    return {
        cuCmul(v.x, s),
        cuCmul(v.y, s),
        cuCmul(v.z, s)
    };
}

__device__ cvec3 add_cvec3(cvec3 a, cvec3 b) {
    return {
        cuCadd(a.x, b.x),
        cuCadd(a.y, b.y),
        cuCadd(a.z, b.z)
    };
}

__device__ cvec3 cross_vec3_cvec3(vec3 a, cvec3 b) {
    return {
        cuCsub(cuCmul(make_cuDoubleComplex(a.y, 0), b.z), cuCmul(make_cuDoubleComplex(a.z, 0), b.y)),
        cuCsub(cuCmul(make_cuDoubleComplex(a.z, 0), b.x), cuCmul(make_cuDoubleComplex(a.x, 0), b.z)),
        cuCsub(cuCmul(make_cuDoubleComplex(a.x, 0), b.y), cuCmul(make_cuDoubleComplex(a.y, 0), b.x))
    };
}

__device__ void print_vec3(const char* name, vec3 v) {
    printf("%s: (%e, %e, %e)\n", name, v.x, v.y, v.z);
}

__device__ void print_cvec3(const char* name, cvec3 v) {
    printf("%s: (%e + %ei, %e + %ei, %e + %ei)\n", 
           name,
           cuCreal(v.x), cuCimag(v.x),
           cuCreal(v.y), cuCimag(v.y),
           cuCreal(v.z), cuCimag(v.z));
}

__device__ void print_complex(const char* name, cuDoubleComplex z) {
    printf("%s: %e + %ei\n", name, cuCreal(z), cuCimag(z));
}

__global__ void compute_field(
    const vec3* dipole_pos, const cvec3* dipole_mom, int N_dip,
    const vec3* obs_pos, cvec3* E_out, cvec3* B_out, int N_obs,
    double k, double prefac)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_obs) return;

    vec3 r_obs = obs_pos[i];
    cvec3 E = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};
    cvec3 B = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    double c = 299792458.0;
    double omega = k * c;
    
    // Only print debug info if this is our special point at (0,0,1e-6)
    bool is_debug_point = false; //(fabs(r_obs.x) < 1e-10 && fabs(r_obs.y) < 1e-10 && fabs(r_obs.z - 1e-6) < 1e-10);
    
    if (is_debug_point) {
        printf("\n=== Computing fields at observation point ===\n");
        print_vec3("Observation point r_obs", r_obs);
        printf("Wave number k: %e\n", k);
        printf("Prefactor: %e\n", prefac);
    }

    for (int j = 0; j < N_dip; ++j) {
        vec3 r_dip = dipole_pos[j];
        cvec3 p = dipole_mom[j];
        
        if (is_debug_point) {
            printf("\n--- Dipole %d contribution ---\n", j);
            print_vec3("Dipole position r_dip", r_dip);
            print_cvec3("Dipole moment p", p);
        }

        vec3 r = device_vec3_sub(r_obs, r_dip);
        double R = device_vec3_norm(r);
        vec3 n = device_vec3_normalize(r);
        
        if (is_debug_point) {
            print_vec3("Distance vector r", r);
            printf("Distance magnitude R: %e\n", R);
            print_vec3("Unit vector n", n);
        }

        // Electric field computation
        cvec3 n_cross_p = cross_vec3_cvec3(n, p);
        if (is_debug_point) print_cvec3("n × p", n_cross_p);
        
        cvec3 term1 = cross_vec3_cvec3(n, n_cross_p);
        term1 = scale_cvec3(term1, make_cuDoubleComplex(-k * k / R, 0.0));
        if (is_debug_point) print_cvec3("Far-field term (k²(n × p) × n)/R", term1);

        cuDoubleComplex n_dot_p = dot_cvec3(n, p);
        if (is_debug_point) print_complex("n · p", n_dot_p);
        
        cvec3 three_n_n_dot_p = {
            cuCmul(make_cuDoubleComplex(3 * n.x, 0.0), n_dot_p),
            cuCmul(make_cuDoubleComplex(3 * n.y, 0.0), n_dot_p),
            cuCmul(make_cuDoubleComplex(3 * n.z, 0.0), n_dot_p),
        };
        if (is_debug_point) print_cvec3("3n(n · p)", three_n_n_dot_p);
        
        cvec3 term2_vec = {
            cuCsub(three_n_n_dot_p.x, p.x),
            cuCsub(three_n_n_dot_p.y, p.y),
            cuCsub(three_n_n_dot_p.z, p.z),
        };
        if (is_debug_point) print_cvec3("3n(n · p) - p", term2_vec);

        cuDoubleComplex scalar = cuCdiv(
            cuCsub(make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, k * R)),
            make_cuDoubleComplex(R * R * R, 0.0)
        );
        if (is_debug_point) print_complex("Near-field scalar (1-ikR)/R³", scalar);
        
        cvec3 term2 = scale_cvec3(term2_vec, scalar);
        if (is_debug_point) print_cvec3("Near-field term [(3n(n · p) - p)(1-ikR)/R³]", term2);

        cuDoubleComplex phase = make_cuDoubleComplex(cos(k * R), sin(k * R));
        if (is_debug_point) print_complex("Phase factor e^(ikR)", phase);
        
        cvec3 total_E = add_cvec3(term1, term2);
        total_E = scale_cvec3(total_E, cuCmul(phase, make_cuDoubleComplex(prefac, 0.0)));
        if (is_debug_point) print_cvec3("Total E field contribution from this dipole", total_E);

        // Magnetic field computation
        double mu0 = 1.25663706212e-6;
        cuDoubleComplex factor = cuCmul(
            make_cuDoubleComplex(0.0, mu0 * omega),
            cuCsub(make_cuDoubleComplex(1.0 / R, 0.0), make_cuDoubleComplex(0.0, k))
        );
        if (is_debug_point) print_complex("B-field factor -iμ₀ω(1/R - ik)", factor);

        cuDoubleComplex scale = cuCdiv(factor, make_cuDoubleComplex(4 * M_PI * R, 0.0));
        scale = cuCmul(scale, phase);
        if (is_debug_point) print_complex("B-field scaling factor", scale);

        cvec3 B_contrib = scale_cvec3(cross_vec3_cvec3(n, p), scale);
        if (is_debug_point) print_cvec3("B field contribution from this dipole", B_contrib);

        E = add_cvec3(E, total_E);
        B = add_cvec3(B, B_contrib);
    }

    if (is_debug_point) {
        printf("\n=== Final Results ===\n");
        print_cvec3("Total E field", E);
        print_cvec3("Total B field", B);
    }
    
    E_out[i] = E;
    B_out[i] = B;
}