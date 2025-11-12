import numpy as np

epsilon_0 = 8.8541878188e-12
mu_0 = 1.25663706127e-6

def calculate_dipole_fields(dipole_location,
                            observation_location,
                            moment_vector,
                            dipole_type,
                            omega):
    """
    Calculates the electric and magnetic fields produced by an electric or magnetic dipole.

    Args:
        dipole_location (list or np.array): 3D coordinates [x, y, z] of the dipole.
        observation_location (list or np.array): 3D coordinates [x, y, z] to calculate fields at.
        moment_vector (list or np.array): 3D dipole moment vector. Interpreted as electric
                                          dipole moment p (in C*m) if dipole_type is 'electric',
                                          or magnetic dipole moment m (in A*m^2) if 'magnetic'.
        dipole_type (str): Type of dipole, either 'electric' or 'magnetic'.
        omega (float): Angular frequency (rad/s) of the dipole oscillation.
        epsilon_0 (float, optional): Permittivity of free space. Defaults to standard value.
        mu_0 (float, optional): Permeability of free space. Defaults to standard value.

    Returns:
        tuple: (E_field, H_field)
            E_field (np.array): Complex 3D vector of the electric field (V/m).
            H_field (np.array): Complex 3D vector of the magnetic field (A/m).
            Returns (nan_vec, nan_vec) if observation point is at the dipole location.
    """

    # --- Setup constants and vectors ---
    c = 1 / np.sqrt(epsilon_0 * mu_0)  # Speed of light in vacuum
    k = omega / c  # Wavenumber

    # Convert inputs to numpy arrays for vector operations
    r_dipole = np.array(dipole_location, dtype=float)
    r_obs = np.array(observation_location, dtype=float)
    moment = np.array(moment_vector, dtype=complex)

    # Calculate displacement vector r, its magnitude, and unit vector
    r_vec = r_obs - r_dipole
    r_mag = np.linalg.norm(r_vec)

    # Handle observation at the dipole location (singularity)
    if r_mag == 0:
        print("Warning: Observation point is at the dipole location. Fields are singular.")
        nan_vec = np.full(3, np.nan, dtype=complex)
        return nan_vec, nan_vec

    r_hat = r_vec / r_mag

    # Common complex exponential factor from Green's function e^(ikr)
    exp_ikr = np.exp(1j * k * r_mag)

    # Initialize E and H fields
    E_field = np.zeros(3, dtype=complex)
    H_field = np.zeros(3, dtype=complex)

    # --- Calculate fields based on dipole type ---
    if dipole_type == 'electric':
        p = moment # Electric dipole moment

        # Electric field E_p(r)
        # E_p(r) = { (1/ε₀)(1/r - ik)(3r̂(r̂·p) - p)/r² + (k²/ε₀)(r̂×(p×r̂))/r } e^(ikr)/(4π)
        # Using r̂×(p×r̂) = p - r̂(r̂·p)
        
        # Term 1: (1/ε₀)(1/r³ - ik/r²)(3r̂(r̂·p) - p)
        term1_coeff_Ep = (1 / (epsilon_0 * r_mag**3)) - (1j * k / (epsilon_0 * r_mag**2))
        term1_vec_Ep = 3 * r_hat * np.dot(r_hat, p) - p
        
        # Term 2: (k²/ε₀r)(p - r̂(r̂·p))
        term2_coeff_Ep = (k**2) / (epsilon_0 * r_mag)
        term2_vec_Ep = p - r_hat * np.dot(r_hat, p)
        
        E_field_curly_braces = term1_coeff_Ep * term1_vec_Ep + term2_coeff_Ep * term2_vec_Ep
        E_field = E_field_curly_braces * (exp_ikr / (4 * np.pi))

        # Magnetic field H_p(r)
        # H_p(r) = -iω(1/r - ik)(p×r̂) e^(ikr)/(4πr)
        coeff_Hp = -1j * omega * (1/r_mag - 1j*k) / (4 * np.pi * r_mag)
        H_field = coeff_Hp * np.cross(p, r_hat) * exp_ikr

    elif dipole_type == 'magnetic':
        m = moment # Magnetic dipole moment

        # Magnetic field H_m(r) - analogous to E_p(r) with p->m and 1/ε₀ factor removed
        # H_m(r) = { (1/r - ik)(3r̂(r̂·m) - m)/r² + k²(r̂×(m×r̂))/r } e^(ikr)/(4π)
        
        # Term 1: (1/r³ - ik/r²)(3r̂(r̂·m) - m)
        term1_coeff_Hm = (1 / r_mag**3) - (1j * k / r_mag**2)
        term1_vec_Hm = 3 * r_hat * np.dot(r_hat, m) - m
        
        # Term 2: (k²/r)(m - r̂(r̂·m))
        term2_coeff_Hm = (k**2) / r_mag
        term2_vec_Hm = m - r_hat * np.dot(r_hat, m)
        
        H_field_curly_braces = term1_coeff_Hm * term1_vec_Hm + term2_coeff_Hm * term2_vec_Hm
        H_field = H_field_curly_braces * (exp_ikr / (4 * np.pi))
        
        # Electric field E_m(r)
        # E_m(r) = iωμ₀(1/r - ik)(m×r̂) e^(ikr)/(4πr)
        coeff_Em = 1j * omega * mu_0 * (1/r_mag - 1j*k) / (4 * np.pi * r_mag)
        E_field = coeff_Em * np.cross(m, r_hat) * exp_ikr
        
    else:
        raise ValueError("dipole_type must be 'electric' or 'magnetic'.")

    return E_field, H_field

def calculate_both_dipole_fields(dipole_location, observation_location, sixmomentvector, omega):
    """
    Convenience function to calculate both electric and magnetic dipole fields.
    
    Args:
        dipole_location (list or np.array): 3D coordinates of the dipole.
        observation_location (list or np.array): 3D coordinates to calculate fields at.
        moment_vector (list or np.array): 3D dipole moment vector.
        omega (float): Angular frequency of the dipole oscillation.
    
    Returns:
        tuple: (E_field, H_field)
    """

    electric_moment_vector, magnetic_moment_vector = np.array(sixmomentvector[:3], dtype=complex), np.array(sixmomentvector[3:], dtype=complex)

    EE_field, EH_field = calculate_dipole_fields(dipole_location, observation_location, electric_moment_vector, 'electric', omega)
    ME_field, MH_field = calculate_dipole_fields(dipole_location, observation_location, magnetic_moment_vector, 'magnetic', omega)
    E_field = EE_field + ME_field
    H_field = EH_field + MH_field

    return np.concatenate((E_field, H_field))

if __name__ == '__main__':
    # --- Example Usage ---
    dipole_loc = [0, 0, 0]
    obs_loc = [0, 0, 1] # Observation point 1m along x-axis
    
    # Example: Electric dipole p = [0, 0, 1e-9] C*m (1 nC*m along z)
    p_moment = [1, 0, 0] 
    freq_hz = 300e6 # 300 MHz
    omega_val = 2 * np.pi * freq_hz

    print(f"--- Electric Dipole Example (p={p_moment}, freq={freq_hz/1e6} MHz) ---")
    E_p, H_p = calculate_dipole_fields(dipole_loc, obs_loc, p_moment, 'electric', omega_val)
    print(f"E_p at {obs_loc} = {E_p} V/m")
    print(f"H_p at {obs_loc} = {H_p} A/m")
    print("-" * 30)

    print(f"--- Magnetic Dipole Example (p={p_moment}, freq={freq_hz/1e6} MHz) ---")
    E_m, H_m = calculate_dipole_fields(dipole_loc, obs_loc, p_moment, 'magnetic', omega_val)
    print(f"E_p at {obs_loc} = {E_m} V/m")
    print(f"H_p at {obs_loc} = {H_m} A/m")
    print("-" * 30)