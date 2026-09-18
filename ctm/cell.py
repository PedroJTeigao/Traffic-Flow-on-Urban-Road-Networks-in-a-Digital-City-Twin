class Link:
    """One road segment (one CTM cell = one OSM link).

    Fixed at construction
    ---------------------
    L  : length (m)                         from lengths[e]
    v  : free-flow speed (m/s)              maxspeed_kmh / 3.6
    w  : backward wave speed (m/s)          3.6 m/s (fixed constant)
    C  : capacity flow (veh/s)              1800 * lanes / 3600
    K  : jam density (veh/m)                1500 * lanes / 1000
    N  : max vehicle storage (veh)          K * L

    State variable — updated every step
    ------------------------------------
    n  : vehicle count (veh),  n ∈ [0, N]
    """

    def __init__(self, link_id, L, v, lanes=1, w=3.6):
        self.id = link_id
        self.L  = L
        self.v  = v
        self.w  = w
        self.C  = 1800 * lanes / 3600    # veh/s
        self.K  = 1500 * lanes / 1000    # veh/m  (1500 veh/km per lane)
        self.N  = self.K * L             # max storage (veh)
        self.n  = 0.0                    # initial condition

    def rho(self):
        """Density  ρ_e = n_e / L_e  (veh/m)."""
        return self.n / self.L

    def D(self):
        """Demand / sending function.
        D_e = min(C_e, v_e * rho_e)
        Maximum flow link e can send downstream.
        """
        return min(self.C, self.v * self.rho())

    def S(self):
        """Supply / receiving function.
        S_e = min(C_e, w_e * (K_e - rho_e))
        Maximum flow link e can accept from upstream.
        """
        return min(self.C, self.w * (self.K - self.rho()))
