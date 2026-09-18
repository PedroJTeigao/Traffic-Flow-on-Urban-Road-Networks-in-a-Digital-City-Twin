"""
Origin-Destination matrix builder for CTM networks.

Generic — works with any spatial demand data, not tied to DeSO or Gothenburg.

Two entry points:
  build_od_dist()  — gravity model with Euclidean distance (m), beta in 1/m
  build_od_time()  — gravity model with shortest-path travel time (s), beta in 1/s
"""

import numpy as np
import networkx as nx


def interpolate_idw(V, centroids, z_values, power=2):
    """Inverse-distance weighting from K zone centroids to N network nodes.

    Parameters
    ----------
    V          : (N, 2)  node positions
    centroids  : (K, 2)  zone centroids
    z_values   : (K,)    values at zone centroids
    power      : IDW exponent (default 2)

    Returns
    -------
    (N,) interpolated values at each node
    """
    V  = np.asarray(V, float)
    C  = np.asarray(centroids, float)
    z  = np.asarray(z_values, float)

    dx = V[:, 0:1] - C[:, 0]      # (N, K)
    dy = V[:, 1:2] - C[:, 1]      # (N, K)
    D  = np.sqrt(dx**2 + dy**2)
    D  = np.maximum(D, 1.0)        # avoid /0 for coincident points

    W = 1.0 / D**power             # (N, K)
    return (W @ z) / W.sum(axis=1)


def _build_graph(E, W):
    G = nx.DiGraph()
    edge_idx = {}
    for i, (u, v) in enumerate(E):
        u, v = int(u), int(v)
        G.add_edge(u, v, weight=float(W[i]))
        edge_idx[(u, v)] = i
    return G, edge_idx


def _accumulate(origins, G, edge_idx, production, attraction, link_flow,
                alpha_raw, threshold, get_cands, get_cost):
    N = len(production)
    for k, i in enumerate(origins):
        if k % 100 == 0:
            print(f"       origin {k}/{len(origins)} …")

        try:
            dist_map, path_map = nx.single_source_dijkstra(G, i)
        except Exception:
            continue

        attr_thresh = attraction.max() * 0.01
        cands = [j for j in get_cands(i, dist_map, path_map)
                 if j != i
                 and attraction[j] >= attr_thresh
                 and j in path_map]

        for j in cands:
            cost_ij = get_cost(i, j, dist_map)
            t_ij    = production[i] * attraction[j] * np.exp(-cost_ij)
            if t_ij < threshold:
                continue

            path = path_map[j]

            for s in range(len(path) - 1):
                eid = edge_idx.get((path[s], path[s + 1]))
                if eid is not None:
                    link_flow[eid] += t_ij

            for s in range(1, len(path) - 1):
                jct   = path[s]
                e_in  = edge_idx.get((path[s - 1], path[s]))
                e_out = edge_idx.get((path[s],     path[s + 1]))
                if e_in is not None and e_out is not None:
                    d = alpha_raw.setdefault(jct, {})
                    d[(e_in, e_out)] = d.get((e_in, e_out), 0.0) + t_ij


def _normalise_alpha(alpha_raw, N):
    alpha = {}
    for node_id, flows in alpha_raw.items():
        in_eids = set(k[0] for k in flows)
        alpha[node_id] = {}
        for e_in in in_eids:
            total = sum(v for (ei, _), v in flows.items() if ei == e_in)
            if total > 0:
                for (ei, eo), v in flows.items():
                    if ei == e_in:
                        alpha[node_id][(ei, eo)] = v / total
    return alpha


def build_od_dist(V, E, weights, centroids, z_prod, z_attr,
                  R_max=800.0, beta=0.002, threshold=1e-4, power=2):
    """OD matrix with Euclidean distance gravity model.

    Parameters
    ----------
    V          : (N, 2)  node positions (x, y)
    E          : (M, 2)  edges as (u, v) node-index pairs
    weights    : (M,)    edge weights for shortest path (travel time, s)
    centroids  : (K, 2)  zone centroids
    z_prod     : (K,)    zone productions (e.g. cars_in_traffic)
    z_attr     : (K,)    zone attractions (e.g. employment)
    R_max      : float   max Euclidean distance between OD pair (m)
    beta       : float   gravity decay (1/m)
    threshold  : float   min T[i][j] to route
    power      : float   IDW exponent

    Returns
    -------
    production, attraction, alpha, link_flow
    """
    V  = np.asarray(V, float)
    E  = np.asarray(E, int)
    W  = np.asarray(weights, float)
    N, M = len(V), len(E)

    print("  [OD-dist] Interpolating demand (IDW) …")
    production = interpolate_idw(V, centroids, z_prod, power)
    attraction = interpolate_idw(V, centroids, z_attr, power)
    print(f"            production range: [{production.min():.2f}, {production.max():.2f}]")
    print(f"            attraction range: [{attraction.min():.2f}, {attraction.max():.2f}]")

    G, edge_idx = _build_graph(E, W)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(V)
        def get_cands(i, dist_map, path_map):
            return tree.query_ball_point(V[i], R_max)
    except ImportError:
        def get_cands(i, dist_map, path_map):
            d = np.hypot(V[:, 0] - V[i, 0], V[:, 1] - V[i, 1])
            return list(np.where(d <= R_max)[0])

    def get_cost(i, j, dist_map):
        return beta * np.hypot(V[j, 0] - V[i, 0], V[j, 1] - V[i, 1])

    link_flow = np.zeros(M)
    alpha_raw = {}
    prod_thresh = production.max() * 0.01
    origins = [i for i in range(N) if production[i] >= prod_thresh]
    print(f"  [OD-dist] Routing {len(origins)} origins (R_max={R_max} m, β={beta} 1/m) …")

    _accumulate(origins, G, edge_idx, production, attraction,
                link_flow, alpha_raw, threshold, get_cands, get_cost)

    print("  [OD-dist] Normalising turning ratios …")
    alpha = _normalise_alpha(alpha_raw, N)
    print(f"  [OD-dist] Done. Junctions with OD alpha: {len(alpha)}/{N} | "
          f"Total OD flow: {link_flow.sum():.1f}")
    return production, attraction, alpha, link_flow


def build_od_time(V, E, weights, centroids, z_prod, z_attr,
                  T_max=180.0, beta=1/900, threshold=1e-4, power=2):
    """OD matrix with shortest-path travel time gravity model.

    Parameters
    ----------
    V          : (N, 2)  node positions (x, y)
    E          : (M, 2)  edges as (u, v) node-index pairs
    weights    : (M,)    edge weights for shortest path (travel time, s)
    centroids  : (K, 2)  zone centroids
    z_prod     : (K,)    zone productions (e.g. cars_in_traffic)
    z_attr     : (K,)    zone attractions (e.g. employment)
    T_max      : float   max shortest-path travel time between OD pair (s)
    beta       : float   gravity decay (1/s) — characteristic time = 1/beta
    threshold  : float   min T[i][j] to route
    power      : float   IDW exponent

    Returns
    -------
    production, attraction, alpha, link_flow
    """
    V  = np.asarray(V, float)
    E  = np.asarray(E, int)
    W  = np.asarray(weights, float)
    N, M = len(V), len(E)

    print("  [OD-time] Interpolating demand (IDW) …")
    production = interpolate_idw(V, centroids, z_prod, power)
    attraction = interpolate_idw(V, centroids, z_attr, power)
    print(f"            production range: [{production.min():.2f}, {production.max():.2f}]")
    print(f"            attraction range: [{attraction.min():.2f}, {attraction.max():.2f}]")

    G, edge_idx = _build_graph(E, W)

    def get_cands(i, dist_map, path_map):
        return [j for j in dist_map if dist_map[j] <= T_max]

    def get_cost(i, j, dist_map):
        return beta * dist_map[j]

    link_flow = np.zeros(M)
    alpha_raw = {}
    prod_thresh = production.max() * 0.01
    origins = [i for i in range(N) if production[i] >= prod_thresh]
    print(f"  [OD-time] Routing {len(origins)} origins "
          f"(T_max={T_max} s, β={beta:.5f} 1/s, t_char={1/beta:.0f} s) …")

    _accumulate(origins, G, edge_idx, production, attraction,
                link_flow, alpha_raw, threshold, get_cands, get_cost)

    print("  [OD-time] Normalising turning ratios …")
    alpha = _normalise_alpha(alpha_raw, N)
    print(f"  [OD-time] Done. Junctions with OD alpha: {len(alpha)}/{N} | "
          f"Total OD flow: {link_flow.sum():.1f}")
    return production, attraction, alpha, link_flow
