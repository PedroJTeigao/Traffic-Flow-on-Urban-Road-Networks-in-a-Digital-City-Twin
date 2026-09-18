from .junction import SourceNode, SinkNode, Junction


class Network:
    """Container for all links and nodes; drives the time loop.

    Parameters
    ----------
    links : dict {link_id: Link}
    nodes : dict {node_id: SourceNode | SinkNode | Junction}
    dt    : float   time step (s), computed by CFL in utils.py
    """

    def __init__(self, links, nodes, dt):
        self.links = links
        self.nodes = nodes
        self.dt    = dt
        self.t     = 0.0

    def step(self):
        """Advance simulation by one time step k -> k+1."""

        # Phase 1: snapshot — READ n, compute D and S, no writes
        D = {lid: link.D() for lid, link in self.links.items()}
        S = {lid: link.S() for lid, link in self.links.items()}

        # Phase 2: node flows — READ D and S, compute flows, no writes to n
        y_in  = {lid: 0.0 for lid in self.links}
        y_out = {lid: 0.0 for lid in self.links}

        for node in self.nodes.values():
            if isinstance(node, SourceNode):
                for lid, flow in node.resolve(self.t, S).items():
                    y_in[lid] = flow

            elif isinstance(node, SinkNode):
                for lid, flow in node.resolve(D).items():
                    y_out[lid] = flow

            elif isinstance(node, Junction):
                y_e, q_out = node.resolve(D, S)
                for lid, flow in y_e.items():
                    y_out[lid] = flow
                for lid, flow in q_out.items():
                    y_in[lid] = flow

        # Phase 3: state update — WRITE n, nothing else
        # n_e(t+dt) = clip[ n_e(t) + dt * (y_in - y_out), 0, N_e ]
        for lid, link in self.links.items():
            link.n = link.n + self.dt * (y_in[lid] - y_out[lid])
            link.n = max(0.0, min(link.n, link.N))

        self.t += self.dt

    def run(self, T):
        """Run simulation for T seconds."""
        steps = int(T / self.dt)
        for _ in range(steps):
            self.step()

    def metrics(self):
        """Per-link output metrics for DTCC visualisation.

        Returns
        -------
        dict {link_id: {'rho', 'vcr', 'speed', 'travel_time', 'n'}}
        """
        out = {}
        for lid, link in self.links.items():
            rho   = link.rho()
            rho_c = link.C / link.v if link.v > 0 else link.K

            if rho <= rho_c:
                speed = link.v
            elif rho > 0:
                speed = link.w * (link.K - rho) / rho
            else:
                speed = link.v
            speed = max(0.0, speed)

            flow = min(link.v * rho, link.w * (link.K - rho))
            flow = max(0.0, flow)
            vcr  = flow / link.C if link.C > 0 else 0.0
            tt   = link.L / speed if speed > 0 else float('inf')

            out[lid] = {
                'rho':         rho,
                'vcr':         vcr,
                'speed':       speed,
                'travel_time': tt,
                'n':           link.n,
            }
        return out
