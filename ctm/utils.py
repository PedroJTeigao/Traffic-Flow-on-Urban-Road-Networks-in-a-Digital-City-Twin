"""
Network construction helpers for traffic_engine_3.

build_network     — identical to traffic_engine/utils.py
from_dtcc         — identical to traffic_engine/utils.py
build_network_od  — builds network with OD pipeline + mid-network absorption
"""

import numpy as np
from .cell     import Link
from .junction import (SourceNode, SinkNode, Junction,
                       InjectionJunction, AbsorptionJunction,
                       InjectionAbsorptionJunction)
from .network  import Network
from .od_matrix import build_od_time
from .merge    import merge_links_into_streets

SPEED_DEFAULTS = {
    'motorway':     110.0,
    'trunk':         90.0,
    'primary':       60.0,
    'secondary':     50.0,
    'tertiary':      40.0,
    'residential':   30.0,
    'unclassified':  30.0,
}
LANES_DEFAULT = 1
SPEED_DEFAULT = 50.0

PRIORITY_WEIGHT = {
    'motorway':       4.0,
    'trunk':          3.5,
    'primary':        3.0,
    'primary_link':   2.5,
    'secondary':      2.0,
    'secondary_link': 1.5,
    'tertiary':       1.5,
    'tertiary_link':  1.2,
    'residential':    1.0,
    'unclassified':   1.0,
}


def build_network(road_data, demand_fns=None, w=3.6, safety=0.9):
    """Build a Network from DTCC / OSM road arrays.

    Parameters
    ----------
    road_data   : dict with keys 'edges', 'lengths', 'maxspeed', 'lanes', 'highway'
    demand_fns  : dict {node_id: callable(t) -> float}  veh/s for source nodes
    w           : float  backward wave speed (m/s)
    safety      : float  CFL safety factor in (0, 1]

    Returns
    -------
    Network
    """
    if demand_fns is None:
        demand_fns = {}

    links = {}
    for _, _, eid in road_data['edges']:
        L = road_data['lengths'][eid]

        if road_data['maxspeed'].get(eid) is not None:
            v_ff = road_data['maxspeed'][eid] / 3.6
        else:
            hw   = road_data.get('highway', {}).get(eid, '')
            v_ff = SPEED_DEFAULTS.get(hw, SPEED_DEFAULT) / 3.6

        lanes = road_data.get('lanes', {}).get(eid, LANES_DEFAULT)
        links[eid] = Link(link_id=eid, L=L, v=v_ff, lanes=lanes, w=w)

    dt = min(
        links[eid].L / max(links[eid].v, links[eid].w)
        for eid in links
    ) * safety

    node_in  = {}
    node_out = {}
    for u, v, eid in road_data['edges']:
        node_out.setdefault(u, []).append(eid)
        node_in.setdefault(v,  []).append(eid)

    nodes = {}
    for nid in set(node_in) | set(node_out):
        in_eids  = node_in.get(nid,  [])
        out_eids = node_out.get(nid, [])

        if not in_eids:
            demand_fn = demand_fns.get(nid, lambda t: 0.0)
            nodes[nid] = SourceNode(out_link_ids=out_eids, demand_fn=demand_fn)

        elif not out_eids:
            nodes[nid] = SinkNode(in_link_ids=in_eids)

        else:
            n_out = len(out_eids)
            alpha = {
                (ein, eout): 1.0 / n_out
                for ein  in in_eids
                for eout in out_eids
            }
            p = {eid: 1.0 / len(in_eids) for eid in in_eids}
            nodes[nid] = Junction(
                in_link_ids=in_eids, out_link_ids=out_eids,
                alpha=alpha, p=p,
            )

    return Network(links=links, nodes=nodes, dt=dt)


def from_dtcc(edges, lengths, maxspeed, lanes, highway=None):
    """Convert DTCC Core arrays to the road_data dict for build_network().

    Parameters
    ----------
    edges    : array-like of (u, v) node-index pairs
    lengths  : array-like of float   (m)
    maxspeed : array-like of float   (km/h); use None for missing values
    lanes    : array-like of int
    highway  : array-like of str     OSM road class (optional)

    Returns
    -------
    road_data dict
    """
    edge_list = [(int(u), int(v), i) for i, (u, v) in enumerate(edges)]
    n = len(edge_list)
    return {
        'edges':    edge_list,
        'lengths':  {i: float(lengths[i])                              for i in range(n)},
        'maxspeed': {i: (float(maxspeed[i]) if maxspeed[i] is not None
                        else None)                                      for i in range(n)},
        'lanes':    {i: int(lanes[i])                                  for i in range(n)},
        'highway':  {i: (str(highway[i]) if highway is not None else '')
                        for i in range(n)},
    }


def build_network_od(road_data, V, centroids, z_prod, z_attr,
                     mode='links',
                     T_max=180.0, demand_scale=1e-5,
                     prod_scale=None, attr_scale=None,
                     high_prod_threshold=0.05,
                     high_attr_threshold=0.05,
                     w=3.6, safety=0.9):
    """Build a Network with OD pipeline and mid-network absorption.

    Each interior Junction with degree >= 3 is classified into one of four
    types based on whether its IDW production and attraction values fall
    above the (1 - high_*_threshold) percentile of all network nodes:

        high production only  → InjectionJunction         (residential)
        high attraction only  → AbsorptionJunction         (employment)
        both high             → InjectionAbsorptionJunction (mixed zone)
        neither               → plain Junction

    injection uses production[nid] * prod_scale,
    absorption uses attraction[nid] * attr_scale.

    Parameters
    ----------
    road_data            : dict  from from_dtcc()
    V                    : (N, 2)  node positions (m)
    centroids            : (K, 2)  DeSO zone centroids
    z_prod               : (K,)    production proxy (cars_in_traffic)
    z_attr               : (K,)    attraction proxy (employed_residents_total)
    mode                 : str     'links' — one CTM cell per OSM edge (default)
                                   'roads' — one CTM cell per street between
                                             real junctions (merged network)
    T_max                : float   max travel time for OD candidates (s)
    demand_scale         : float   fallback scale for both prod and attr (veh/s)
    prod_scale           : float   injection scale; overrides demand_scale if set
    attr_scale           : float   absorption scale; overrides demand_scale if set
    high_prod_threshold  : float   top fraction of nodes classified as high
                                   production; 0.05 → 95th percentile
    high_attr_threshold  : float   same for attraction
    w                    : float   backward wave speed (m/s)
    safety               : float   CFL safety factor

    Returns
    -------
    Network
        Extra attributes set on the returned object:
          net.mode               — 'links' or 'roads'
          net.street_to_links    — {new_eid: [orig_eids]} or None
          net.junction_positions — {node_id: (x,y)} or None
          net.edges              — [(u, v, eid), …] list used to build net
    """
    V = np.asarray(V, float)

    # ── resolve prod/attr scales ───────────────────────────────────────────────
    _prod_scale = prod_scale if prod_scale is not None else demand_scale
    _attr_scale = attr_scale if attr_scale is not None else demand_scale

    # ── 0. mode: optionally merge degree-2 nodes into streets ─────────────────
    if mode == 'roads':
        road_data, street_to_links, junction_positions = \
            merge_links_into_streets(road_data, V)
        print(f"  [mode=roads] Merged to {len(road_data['edges'])} streets")
    elif mode == 'links':
        street_to_links    = None
        junction_positions = None
    else:
        raise ValueError(f"mode must be 'links' or 'roads', got {mode!r}")

    # ── 1. edge array and travel-time weights ─────────────────────────────────
    E       = np.array([(u, v) for u, v, _ in road_data['edges']], dtype=int)
    n_edges = len(E)

    speeds_ms = np.array([
        (road_data['maxspeed'][i] / 3.6
         if road_data['maxspeed'].get(i) is not None
         else SPEED_DEFAULTS.get(road_data.get('highway', {}).get(i, ''),
                                 SPEED_DEFAULT) / 3.6)
        for i in range(n_edges)
    ])
    lengths = np.array([road_data['lengths'][i] for i in range(n_edges)])
    tt_edge = lengths / np.maximum(speeds_ms, 0.1)

    # ── 2. beta calibration ───────────────────────────────────────────────────
    cen = np.asarray(centroids, float)
    _d  = np.sqrt(((cen[:, None, :] - cen[None, :, :])**2).sum(-1))
    _mean_dist  = _d[_d > 0].mean()
    _mean_speed = float(speeds_ms.mean())
    beta = _mean_speed / _mean_dist
    print(f"  [build_network_od] β = {beta:.6f} 1/s  "
          f"(mean travel time ≈ {1/beta:.1f} s = {1/beta/60:.1f} min)")

    # ── 3. OD matrix ──────────────────────────────────────────────────────────
    production, attraction, alpha_by_node, link_flow = build_od_time(
        V         = V,
        E         = E,
        weights   = tt_edge,
        centroids = cen,
        z_prod    = np.asarray(z_prod, float),
        z_attr    = np.asarray(z_attr, float),
        T_max     = T_max,
        beta      = beta,
        threshold = 1e-4,
        power     = 2,
    )
    print(f"  [build_network_od] Total OD flow: {link_flow.sum():.1f}  |  "
          f"Junctions with OD alpha: {len(alpha_by_node)}")

    # ── 4. demand_fns for SourceNodes ─────────────────────────────────────────
    node_in  = {}
    node_out = {}
    for u, v, eid in road_data['edges']:
        node_out.setdefault(u, []).append(eid)
        node_in.setdefault(v,  []).append(eid)

    demand_fns = {}
    for nid in set(node_out) - set(node_in):
        if nid < len(production) and production[nid] > 0:
            rate = float(production[nid]) * _prod_scale
            demand_fns[nid] = lambda t, r=rate: r

    # ── 5. build base network ─────────────────────────────────────────────────
    net = build_network(road_data, demand_fns=demand_fns, w=w, safety=safety)

    # ── 6. percentile thresholds ──────────────────────────────────────────────
    prod_thresh = float(np.percentile(production, (1 - high_prod_threshold) * 100))
    attr_thresh = float(np.percentile(attraction, (1 - high_attr_threshold) * 100))

    counts = {'injection': 0, 'absorption': 0,
              'injection_absorption': 0, 'plain': 0}

    hw_map = road_data.get('highway', {})

    for nid, nd in list(net.nodes.items()):
        if not isinstance(nd, Junction):
            continue

        in_eids  = nd.in_link_ids
        out_eids = nd.out_link_ids

        # ── patch OD alpha ────────────────────────────────────────────────────
        if nid in alpha_by_node:
            od_pairs  = alpha_by_node[nid]
            new_alpha = dict(nd.alpha)
            for e_in in in_eids:
                row_sum = sum(v for (ei, _), v in od_pairs.items() if ei == e_in)
                if row_sum == 0:
                    continue
                row = {}
                for e_out in out_eids:
                    row[(e_in, e_out)] = od_pairs.get((e_in, e_out), 1e-6)
                row_total = sum(row.values())
                for key in row:
                    new_alpha[key] = row[key] / row_total
            nd.alpha = new_alpha

        # ── highway priority ──────────────────────────────────────────────────
        pw       = {eid: PRIORITY_WEIGHT.get(hw_map.get(eid, ''), 1.0)
                    for eid in in_eids}
        total_pw = sum(pw.values())
        nd.p     = {eid: pw[eid] / total_pw for eid in in_eids}

        # ── classify ──────────────────────────────────────────────────────────
        is_real      = (len(in_eids) + len(out_eids)) >= 3
        is_high_prod = is_real and nid < len(production) and production[nid] >= prod_thresh
        is_high_attr = is_real and nid < len(attraction) and attraction[nid] >= attr_thresh

        n_out           = len(out_eids)
        od_row          = alpha_by_node.get(nid, {})
        absorption_rate = float(attraction[nid]) * _attr_scale if nid < len(attraction) else 0.0

        extra_inflow = {}
        if is_high_prod:
            rate = float(production[nid]) * _prod_scale
            for fid in out_eids:
                alpha_w = max(
                    (v for (ei, eo), v in od_row.items() if eo == fid),
                    default=1.0 / n_out,
                )
                extra_inflow[fid] = rate * alpha_w

        if is_high_prod and is_high_attr:
            net.nodes[nid] = InjectionAbsorptionJunction(
                in_link_ids     = in_eids,
                out_link_ids    = out_eids,
                alpha           = nd.alpha,
                p               = nd.p,
                extra_inflow    = extra_inflow,
                absorption_rate = absorption_rate,
            )
            counts['injection_absorption'] += 1

        elif is_high_prod:
            net.nodes[nid] = InjectionJunction(
                in_link_ids  = in_eids,
                out_link_ids = out_eids,
                alpha        = nd.alpha,
                p            = nd.p,
                extra_inflow = extra_inflow,
            )
            counts['injection'] += 1

        elif is_high_attr:
            net.nodes[nid] = AbsorptionJunction(
                in_link_ids     = in_eids,
                out_link_ids    = out_eids,
                alpha           = nd.alpha,
                p               = nd.p,
                absorption_rate = absorption_rate,
            )
            counts['absorption'] += 1

        else:
            counts['plain'] += 1

    print(f"  [build_network_od] Node types: "
          f"{counts['injection']} InjectionJunction | "
          f"{counts['absorption']} AbsorptionJunction | "
          f"{counts['injection_absorption']} InjectionAbsorptionJunction | "
          f"{counts['plain']} plain Junction")

    net.mode               = mode
    net.street_to_links    = street_to_links
    net.junction_positions = junction_positions
    net.edges              = road_data['edges']
    return net
