"""
Node types for the CTM network.

SourceNode                — network entry point (no incoming edges)
SinkNode                  — network exit point  (no outgoing edges)
Junction                  — general N-in / M-out intersection (Lebacque & Khoshyaran 2002)
InjectionJunction         — Junction that injects local production demand
AbsorptionJunction        — Junction that absorbs arriving vehicles (destination)
InjectionAbsorptionJunction — Junction that both injects and absorbs (mixed zone)
"""


class SourceNode:
    """Entry point — injects external demand b_e(t) into the network.

    Parameters
    ----------
    out_link_ids : list[int]
    demand_fn    : callable(t) -> float   veh/s  (from DeSO schedule)
    """

    def __init__(self, out_link_ids, demand_fn):
        self.out_link_ids = out_link_ids
        self.demand_fn    = demand_fn

    def resolve(self, t, S):
        """Return inflow for each outgoing link.

        y_in,e(t) = min{ S_e(n_e), b_e(t) }

        Parameters
        ----------
        t : float  current simulation time (s)
        S : dict   {link_id: supply value}   from phase-1 snapshot

        Returns
        -------
        dict {link_id: inflow (veh/s)}
        """
        b = self.demand_fn(t)
        return {fid: min(b, S[fid]) for fid in self.out_link_ids}


class SinkNode:
    """Exit point — absorbs all vehicles that reach the network boundary.

    Parameters
    ----------
    in_link_ids : list[int]
    """

    def __init__(self, in_link_ids):
        self.in_link_ids = in_link_ids

    def resolve(self, D):
        """Return outflow from each incoming link.

        y_out,e(t) = D_e(t)   (no downstream constraint at sink)

        Parameters
        ----------
        D : dict {link_id: demand value}   from phase-1 snapshot

        Returns
        -------
        dict {link_id: outflow (veh/s)}
        """
        return {eid: D[eid] for eid in self.in_link_ids}


class Junction:
    """General N-in / M-out intersection node (Lebacque & Khoshyaran 2002).

    Parameters
    ----------
    in_link_ids  : list[id]
    out_link_ids : list[id]
    alpha        : dict {(e_id, f_id): float}
        Turning ratios. Must satisfy sum_f alpha[e,f] = 1 for each e.
    p            : dict {e_id: float}  (optional)
        Priority weights. Default = equal weight for all incoming links.
    """

    def __init__(self, in_link_ids, out_link_ids, alpha, p=None):
        self.in_link_ids  = in_link_ids
        self.out_link_ids = out_link_ids
        self.alpha        = alpha
        if p is None:
            w = 1.0 / max(len(in_link_ids), 1)
            p = {eid: w for eid in in_link_ids}
        self.p = p

    def resolve(self, D, S):
        """Compute junction flows (Lebacque & Khoshyaran 2002).

        Parameters
        ----------
        D : dict {link_id: demand}   phase-1 snapshot
        S : dict {link_id: supply}   phase-1 snapshot

        Returns
        -------
        y_e   : dict {e_id: y_e}    total outflow from each incoming link
        q_out : dict {f_id: q_f}    total inflow  into each outgoing link
        """
        in_ids  = self.in_link_ids
        out_ids = self.out_link_ids

        # per-link supply cap: mu[e] = min_{f: alpha[e,f] > 0}( S[f] / alpha[e,f] )
        mu = {}
        for eid in in_ids:
            caps = [
                S[fid] / self.alpha[(eid, fid)]
                for fid in out_ids
                if self.alpha.get((eid, fid), 0.0) > 0.0
            ]
            mu[eid] = min(caps) if caps else D[eid]

        # global throttle: Y = min_f( S[f] / sum_e(alpha[e,f] * p[e]) )
        throttles = []
        for fid in out_ids:
            denom = sum(
                self.alpha.get((eid, fid), 0.0) * self.p.get(eid, 0.0)
                for eid in in_ids
            )
            if denom > 0.0:
                throttles.append(S[fid] / denom)
        Y = min(throttles) if throttles else float('inf')

        # y_e = min( D_e, mu[e], p_e * Y )
        y_e = {
            eid: min(D[eid], mu[eid], self.p.get(eid, 1.0) * Y)
            for eid in in_ids
        }

        # q_ef = alpha[e,f] * y_e  →  q_f = sum_e q_ef
        q_out = {fid: 0.0 for fid in out_ids}
        for eid in in_ids:
            for fid in out_ids:
                q_out[fid] += self.alpha.get((eid, fid), 0.0) * y_e[eid]

        return y_e, q_out


class InjectionJunction(Junction):
    """Junction that also injects local OD demand into outgoing links.

    After standard Lebacque resolution, extra flow is added to each outgoing
    link proportional to the local production rate, capped by remaining supply.

    Parameters
    ----------
    in_link_ids   : list[int]
    out_link_ids  : list[int]
    alpha         : dict {(e_id, f_id): float}
    p             : dict {e_id: float}  optional
    extra_inflow  : dict {link_id: float}  additional injection rate (veh/s)
    """

    def __init__(self, in_link_ids, out_link_ids, alpha, p=None,
                 extra_inflow=None):
        super().__init__(in_link_ids, out_link_ids, alpha, p)
        self.extra_inflow   = extra_inflow or {}
        self.injected_this_step = 0.0

    def resolve(self, D, S):
        """Run standard junction logic then inject extra demand.

        Returns
        -------
        y_e   : dict {e_id: float}
        q_out : dict {f_id: float}  includes extra injection
        """
        y_e, q_out = super().resolve(D, S)
        self.injected_this_step = 0.0

        for fid, extra in self.extra_inflow.items():
            if fid in q_out and extra > 0.0:
                remaining = max(S[fid] - q_out[fid], 0.0)
                added = min(extra, remaining)
                q_out[fid] += added
                self.injected_this_step += added

        return y_e, q_out


class AbsorptionJunction(Junction):
    """Junction that absorbs a fraction of arriving vehicles (destination node).

    After standard Lebacque resolution, outgoing flows are scaled down
    uniformly. The absorbed flow represents vehicles reaching their destination
    and leaving the simulation mid-network.

    The absorbed fraction is:
        fraction = min(1, absorption_rate / total_inflow)

    so absorption is always bounded by what physically arrives.

    Parameters
    ----------
    in_link_ids    : list[int]
    out_link_ids   : list[int]
    alpha          : dict {(e_id, f_id): float}
    p              : dict {e_id: float}  optional
    absorption_rate : float  desired absorption flow (veh/s)
    """

    def __init__(self, in_link_ids, out_link_ids, alpha, p=None,
                 absorption_rate=0.0):
        super().__init__(in_link_ids, out_link_ids, alpha, p)
        self.absorption_rate    = absorption_rate
        self.absorbed_this_step = 0.0

    def resolve(self, D, S):
        """Run standard junction logic then absorb a fraction of inflow.

        Returns
        -------
        y_e   : dict {e_id: float}  unchanged — incoming links drain normally
        q_out : dict {f_id: float}  scaled down by absorption fraction
        """
        y_e, q_out = super().resolve(D, S)

        total_inflow = sum(y_e.values())
        self.absorbed_this_step = 0.0

        if total_inflow > 0.0:
            fraction = min(1.0, self.absorption_rate / total_inflow)
            absorbed = total_inflow * fraction
            q_out = {fid: v * (1.0 - fraction) for fid, v in q_out.items()}
            self.absorbed_this_step = absorbed

        return y_e, q_out


class InjectionAbsorptionJunction(Junction):
    """Junction that both injects local demand and absorbs arriving vehicles.

    Used for mixed zones (residential + employment) where cars both depart
    from and arrive at the same node.

    Order: Lebacque → absorb → inject.
    Absorbing before injecting ensures newly injected vehicles are not
    immediately absorbed in the same time step.

    Parameters
    ----------
    in_link_ids     : list[int]
    out_link_ids    : list[int]
    alpha           : dict {(e_id, f_id): float}
    p               : dict {e_id: float}  optional
    extra_inflow    : dict {link_id: float}  injection rate per outgoing link (veh/s)
    absorption_rate : float  desired absorption flow (veh/s)
    """

    def __init__(self, in_link_ids, out_link_ids, alpha, p=None,
                 extra_inflow=None, absorption_rate=0.0):
        super().__init__(in_link_ids, out_link_ids, alpha, p)
        self.extra_inflow       = extra_inflow or {}
        self.absorption_rate    = absorption_rate
        self.absorbed_this_step = 0.0
        self.injected_this_step = 0.0

    def resolve(self, D, S):
        """Lebacque → absorb → inject.

        Returns
        -------
        y_e   : dict {e_id: float}
        q_out : dict {f_id: float}
        """
        y_e, q_out = super().resolve(D, S)

        # ── absorb ────────────────────────────────────────────────────────────
        total_inflow = sum(y_e.values())
        self.absorbed_this_step = 0.0

        if total_inflow > 0.0:
            fraction = min(1.0, self.absorption_rate / total_inflow)
            q_out = {fid: v * (1.0 - fraction) for fid, v in q_out.items()}
            self.absorbed_this_step = total_inflow * fraction

        # ── inject ────────────────────────────────────────────────────────────
        self.injected_this_step = 0.0

        for fid, extra in self.extra_inflow.items():
            if fid in q_out and extra > 0.0:
                remaining = max(S[fid] - q_out[fid], 0.0)
                added = min(extra, remaining)
                q_out[fid] += added
                self.injected_this_step += added

        return y_e, q_out
