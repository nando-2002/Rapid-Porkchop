import math
from numba import cuda, types

@cuda.jit(device=True)
def asin_dev(x):
    """Safe arcsin for CUDA with bounds checking."""
    if x > 1.0:
        x = 1.0
    if x < -1.0:
        x = -1.0
    return math.asin(x)

@cuda.jit(device=True)
def compute_rp(delta, v_inf_minus, v_inf_plus, mu, rp_guess, max_iter=50, tol=1e-8):
    """Newton-Raphson solver for periapsis radius in powered flyby."""
    rp = rp_guess
    
    for iter in range(max_iter):
        # Incoming hyperbola
        e_minus = 1.0 + (rp * v_inf_minus * v_inf_minus) / mu
        if e_minus <= 1.0:
            return -1.0
        
        delta_minus = 2.0 * asin_dev(1.0 / e_minus)
        
        # Outgoing hyperbola
        e_plus = 1.0 + (rp * v_inf_plus * v_inf_plus) / mu
        if e_plus <= 1.0:
            return -1.0
        
        delta_plus = 2.0 * asin_dev(1.0 / e_plus)
        
        # Residual: δ = (δ⁻ + δ⁺)/2
        f = 0.5 * (delta_minus + delta_plus) - delta
        
        # Derivatives for Newton step
        de_minus_dr = v_inf_minus * v_inf_minus / mu
        de_plus_dr = v_inf_plus * v_inf_plus / mu
        
        d_delta_minus_de = -1.0 / math.sqrt(e_minus*e_minus - 1.0)
        d_delta_plus_de = -1.0 / math.sqrt(e_plus*e_plus - 1.0)
        
        df_dr = 0.5 * (d_delta_minus_de * de_minus_dr + d_delta_plus_de * de_plus_dr)
        
        if abs(df_dr) < 1e-12:
            break
            
        rp_new = rp - f / df_dr
        
        # Bounds check (Earth example: >6571km = 6371+200km alt)
        if rp_new < 3900.0 or rp_new > 1e6:
            return -1.0
            
        if abs(rp_new - rp) < tol:
            break
            
        rp = rp_new
    
    return rp

@cuda.jit(device=True)
def powered_flyby_dv(v_inf_minus_x, v_inf_minus_y, v_inf_minus_z,
                     v_inf_plus_x, v_inf_plus_y, v_inf_plus_z,
                     mu_planet):
    """Compute powered flyby ΔV at periapsis.

    Expects component inputs where:
      - v_inf_minus_* = (spacecraft heliocentric incoming) - (planet heliocentric)
      - v_inf_plus_*  = (spacecraft heliocentric outgoing)  - (planet heliocentric)
    """
    # Excess speed magnitudes
    v_inf_minus = math.sqrt(v_inf_minus_x*v_inf_minus_x + 
                           v_inf_minus_y*v_inf_minus_y + 
                           v_inf_minus_z*v_inf_minus_z)
    v_inf_plus = math.sqrt(v_inf_plus_x*v_inf_plus_x + 
                          v_inf_plus_y*v_inf_plus_y + 
                          v_inf_plus_z*v_inf_plus_z)
    
    if v_inf_minus < 1e-6 or v_inf_plus < 1e-6:
        return 99999.0
    
    # Total turning angle δ
    dot = (v_inf_minus_x * v_inf_plus_x + 
           v_inf_minus_y * v_inf_plus_y + 
           v_inf_minus_z * v_inf_plus_z)
    cos_delta = dot / (v_inf_minus * v_inf_plus)
    if cos_delta > 1.0: cos_delta = 1.0
    if cos_delta < -1.0: cos_delta = -1.0
    delta = math.acos(cos_delta)
    
    # Solve for common periapsis radius
    rp_guess = 8000.0
    rp = compute_rp(delta, v_inf_minus, v_inf_plus, mu_planet, rp_guess)
    
    if rp < 0:
        return 99999.0
    
    # Periapsis speeds for both hyperbolas
    e_minus = 1.0 + (rp * v_inf_minus * v_inf_minus) / mu_planet
    a_minus = -rp / (2.0 * (1.0 - 1.0/e_minus))
    v_p_minus = math.sqrt(mu_planet * (2.0/rp + 1.0/a_minus))
    
    e_plus = 1.0 + (rp * v_inf_plus * v_inf_plus) / mu_planet
    a_plus = -rp / (2.0 * (1.0 - 1.0/e_plus))
    v_p_plus = math.sqrt(mu_planet * (2.0/rp + 1.0/a_plus))
    
    return abs(v_p_plus - v_p_minus)
