import numpy as np
import jax
import jax.numpy as jnp
import scipy.stats
import scipy.integrate
from functools import partial

def _fibonacci_hemisphere(n_pts):
    """Generates a nearly uniform grid of candidate zone axes on the upper hemisphere."""
    i = np.arange(n_pts * 2) + 0.5
    phi = np.arccos(1.0 - i / n_pts)  # 0 to pi/2
    theta = np.pi * (1.0 + 5.0**0.5) * i
    pts = np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)], axis=1)
    return pts[pts[:, 2] >= 0][:n_pts]

@partial(jax.jit, static_argnames=["chunk_size"])
def compute_zone_field(q, grid, gamma_val, nu_val, chunk_size=10000):
    """
    Computes the cross-product pairs of all events and evaluates the heavy-tailed 
    Holtsmark/Moffat field across the candidate zone-axis grid.
    Vectorized using jax.lax.scan to prevent OOM errors on the massive O(N^2) pairs.
    """
    N = q.shape[0]
    # Compute all unique pairs (upper triangle)
    i, j = jnp.triu_indices(N, k=1)
    n_ij = jnp.cross(q[i], q[j])
    
    # Normalize pair vectors (zone axis normals)
    norm = jnp.linalg.norm(n_ij, axis=1, keepdims=True)
    valid = (norm[:, 0] > 1e-5)  # Drop exactly collinear events
    n_ij = jnp.where(valid[:, None], n_ij / jnp.where(norm == 0, 1.0, norm), 0.0)
    
    P = n_ij.shape[0]
    n_chunks = (P + chunk_size - 1) // chunk_size
    pad_len = n_chunks * chunk_size - P
    
    # Pad to perfectly match the chunk boundaries
    n_ij_pad = jnp.pad(n_ij, ((0, pad_len), (0, 0)))
    valid_pad = jnp.pad(valid, ((0, pad_len),))
    
    n_ij_reshaped = n_ij_pad.reshape((n_chunks, chunk_size, 3))
    valid_reshaped = valid_pad.reshape((n_chunks, chunk_size))
    
    def scan_body(carry, carry_args):
        pairs, vmask = carry_args
        # Absolute dot product because +n_ij and -n_ij define the exact same zone axis
        cos_theta = jnp.abs(jnp.dot(pairs, grid.T)) 
        
        # Chordal distance on the hemisphere: D^2 = 2*(1 - cos(theta))
        D_sq = 2.0 * (1.0 - cos_theta)
        
        # Heavy-Tailed Kernel evaluation (Moffat/Student-t)
        k_val = (1.0 + D_sq / (gamma_val**2)) ** (-nu_val)
        
        # Mask out invalid/padded pairs
        k_val = k_val * vmask[:, None]
        
        # Accumulate the field intensity across the grid
        return carry + jnp.sum(k_val, axis=0), None
        
    field, _ = jax.lax.scan(scan_body, jnp.zeros(grid.shape[0], dtype=jnp.result_type(float)), (n_ij_reshaped, valid_reshaped))
    return field, jnp.sum(valid)


class PairwiseSparsifier:
    """
    Global Capture mechanism. Exploits the Product Poisson Point Process of random 
    event pairs on the unit sphere to deterministically isolate Zone Axes using 
    Campbell's Theorem and Banach-space heavy-tailed fields.
    """
    def __init__(self, target_fp=1.0, gamma=0.05, nu=1.5, n_grid=10000):
        self.target_fp = target_fp
        self.gamma = gamma
        self.nu = nu
        self.n_grid = n_grid
        self.grid = _fibonacci_hemisphere(n_grid)
        
        # Precompute the true analytical Gamma approximation integrals for the Hemisphere
        I1, I2 = self._kernel_integrals(gamma, nu)
        self.I1 = I1
        self.I2 = I2
        
    @staticmethod
    def _kernel_integrals(gamma, nu):
        """
        Integrates the kernel over the hemisphere to determine the exact Campbell 
        cumulants. Since cos(theta) goes from 0 to 1, D^2 = 2(1 - cos(theta)).
        Let x = 1 - cos(theta). The solid angle differential d(Omega) integrates to 2pi * dx.
        """
        def k1(x): return (1.0 + 2.0*x / gamma**2)**(-nu)
        def k2(x): return (1.0 + 2.0*x / gamma**2)**(-2*nu)
        
        I1, _ = scipy.integrate.quad(k1, 0.0, 1.0)
        I2, _ = scipy.integrate.quad(k2, 0.0, 1.0)
        
        return 2 * np.pi * I1, 2 * np.pi * I2
        
    def find_zone_axes(self, q_unit):
        """
        Takes a highly sparsified array of N observed unit vectors.
        Returns the dominant orthogonal Zone Axes extracted from the O(N^2) product space.
        """
        grid_jax = jax.device_put(self.grid)
        q_jax = jax.device_put(q_unit)
        
        # 1. Evaluate heavy-tailed field natively on GPU
        field, P_valid = compute_zone_field(q_jax, grid_jax, self.gamma, self.nu)
        P_valid = float(P_valid)
        field = np.array(field)
        
        if P_valid < 10:
            return np.zeros((0, 3)), field, 0.0
            
        # 2. Derive the exact Campbell Null-Hypothesis Threshold
        lam = P_valid / (2 * np.pi)  # Density of pair normals on the hemisphere
        kappa_1 = lam * self.I1
        kappa_2 = lam * self.I2
        
        theta = kappa_2 / kappa_1
        k_gamma = kappa_1 / theta
        
        percentile = 1.0 - (self.target_fp / self.n_grid)
        percentile = min(max(percentile, 0.0), 1.0 - 1e-15)
        
        # Reverse-engineer the Holtsmark field limit using the Gamma distribution
        threshold = scipy.stats.gamma.ppf(percentile, a=k_gamma, scale=theta)
        
        # 3. Extract verified Zone Axes
        mask = field > threshold
        zone_axes = self.grid[mask]
        
        return zone_axes, field, threshold
