"""
Street merging pre-processor for traffic_engine_3.

Collapses chains of degree-2 nodes into single links so that the
Network has one CTM cell per street segment between real junctions,
rather than one cell per OSM edge.
"""

from collections import defaultdict
import numpy as np


def merge_links_into_streets(road_data, V):
    """Collapse chains of degree-2 nodes into single street links.

    A *degree-2 node* touches exactly two distinct undirected edges and
    has no branching — it is a pass-through point on a straight road
    segment.  Collapsing these nodes reduces the network to one CTM cell
    per street between real junctions (intersections, dead-ends).

    Each direction of a bidirectional road is treated independently, so
    two anti-parallel merged streets may share the same pair of junction
    endpoints.

    Edge cases handled:
    - Self-loops (u == v): kept as-is.
    - Cycles of degree-2 nodes (no real junction reachable): kept as-is.

    Parameters
    ----------
    road_data : dict
        Keys: 'edges'    list[(u, v, eid)],
              'lengths'  {eid: float}  (m),
              'maxspeed' {eid: float|None}  (km/h),
              'lanes'    {eid: int},
              'highway'  {eid: str}
    V : array-like, shape (N, 2)
        Node positions (x, y) in metres.

    Returns
    -------
    merged_road_data : dict
        Same structure as road_data with merged links.
        New edge IDs are sequential: 0, 1, …, n_streets-1.
    street_to_links : dict {new_eid: list[original_eid]}
        Maps each merged street id to the ordered list of OSM edge ids
        it was built from.
    junction_positions : dict {node_id: (float, float)}
        (x, y) positions of all real junction nodes (endpoints of merged
        streets), keyed by original node id.
    """
    edges    = road_data['edges']
    lengths  = road_data['lengths']
    maxspeed = road_data['maxspeed']
    lanes    = road_data.get('lanes', {})
    highway  = road_data.get('highway', {})

    # ── 1. undirected degree ──────────────────────────────────────────────────
    undirected_nbrs = defaultdict(set)
    for u, v, eid in edges:
        if u != v:
            undirected_nbrs[u].add(v)
            undirected_nbrs[v].add(u)

    deg = {nid: len(nbrs) for nid, nbrs in undirected_nbrs.items()}
    for u, v, _ in edges:
        for nid in (u, v):
            if nid not in deg:
                deg[nid] = 0

    def is_junction(nid):
        """True when nid is a real junction (not a degree-2 pass-through)."""
        return deg.get(nid, 0) != 2

    # ── 2. directed out-adjacency ─────────────────────────────────────────────
    directed_out = defaultdict(list)
    for u, v, eid in edges:
        directed_out[u].append((v, eid))

    # ── 3. chain traversal from junction nodes ────────────────────────────────
    visited_eids    = set()
    new_edges       = []
    new_lengths     = {}
    new_maxspeed    = {}
    new_lanes       = {}
    new_highway     = {}
    street_to_links = {}
    new_eid         = 0

    for u, v, eid in edges:
        if eid in visited_eids:
            continue

        # self-loop: keep as-is
        if u == v:
            visited_eids.add(eid)
            _store(new_eid, u, v, [eid],
                   new_edges, new_lengths, new_maxspeed, new_lanes, new_highway,
                   street_to_links, lengths, maxspeed, lanes, highway)
            new_eid += 1
            continue

        # only start new chains from real junction nodes
        if not is_junction(u):
            continue

        chain_nodes, chain_eids = _follow_chain(
            u, v, eid, directed_out, deg
        )
        for e in chain_eids:
            visited_eids.add(e)

        _store(new_eid, chain_nodes[0], chain_nodes[-1], chain_eids,
               new_edges, new_lengths, new_maxspeed, new_lanes, new_highway,
               street_to_links, lengths, maxspeed, lanes, highway)
        new_eid += 1

    # ── 4. unvisited edges: degree-2 cycles ───────────────────────────────────
    for u, v, eid in edges:
        if eid in visited_eids:
            continue
        visited_eids.add(eid)
        _store(new_eid, u, v, [eid],
               new_edges, new_lengths, new_maxspeed, new_lanes, new_highway,
               street_to_links, lengths, maxspeed, lanes, highway)
        new_eid += 1

    # ── 5. junction positions ─────────────────────────────────────────────────
    V_arr = np.asarray(V, float)
    junction_ids = {u for u, v, _ in new_edges} | {v for u, v, _ in new_edges}
    junction_positions = {}
    for nid in junction_ids:
        if 0 <= nid < len(V_arr):
            junction_positions[nid] = (float(V_arr[nid, 0]), float(V_arr[nid, 1]))

    merged_road_data = {
        'edges':    new_edges,
        'lengths':  new_lengths,
        'maxspeed': new_maxspeed,
        'lanes':    new_lanes,
        'highway':  new_highway,
    }
    return merged_road_data, street_to_links, junction_positions


# ── helpers ───────────────────────────────────────────────────────────────────

def _follow_chain(start_u, start_v, start_eid, directed_out, deg):
    """Follow a directed edge chain through degree-2 nodes.

    Starting from the directed edge start_u → start_v, advance through
    consecutive degree-2 nodes until a real junction is reached, a fork
    is encountered, or a cycle is detected.

    Parameters
    ----------
    start_u, start_v : int
    start_eid        : int
    directed_out     : dict {u: [(v, eid), …]}
    deg              : dict {nid: undirected_degree}

    Returns
    -------
    chain_nodes : list[int]   node sequence, length >= 2
    chain_eids  : list[int]   edge sequence, length = len(chain_nodes) - 1
    """
    chain_nodes = [start_u, start_v]
    chain_eids  = [start_eid]
    prev, cur   = start_u, start_v

    while True:
        if deg.get(cur, 0) != 2:
            break
        next_edges = [(v, e) for v, e in directed_out.get(cur, []) if v != prev]
        if len(next_edges) != 1:
            break
        next_v, next_eid = next_edges[0]
        if next_v in chain_nodes:
            break
        chain_nodes.append(next_v)
        chain_eids.append(next_eid)
        prev, cur = cur, next_v

    return chain_nodes, chain_eids


def _merge_attrs(chain_eids, lengths, maxspeed, lanes, highway):
    """Compute merged link attributes from a chain of original edges.

    Speed uses the length-weighted harmonic mean so that the merged link
    has the same free-flow travel time as the original chain:

        v_eff = Σ L_i  /  Σ (L_i / v_i)

    Lane count uses the minimum (bottleneck capacity).
    Highway type is taken from the first edge in the chain.

    Parameters
    ----------
    chain_eids : list[int]

    Returns
    -------
    total_length  : float   (m)
    eff_speed_kmh : float|None   (km/h), None if all speeds missing
    min_lanes     : int
    first_hw      : str
    """
    SPEED_DEFAULTS = {
        'motorway': 110.0, 'trunk': 90.0, 'primary': 60.0,
        'secondary': 50.0, 'tertiary': 40.0,
        'residential': 30.0, 'unclassified': 30.0,
    }
    DEFAULT_SPEED = 50.0

    total_length  = 0.0
    total_tt      = 0.0
    min_lanes_val = None
    first_hw      = highway.get(chain_eids[0], '')
    has_speed     = False

    for eid in chain_eids:
        L   = lengths[eid]
        total_length += L

        raw = maxspeed.get(eid)
        if raw is not None:
            v_kmh     = float(raw)
            has_speed = True
        else:
            hw    = highway.get(eid, '')
            v_kmh = SPEED_DEFAULTS.get(hw, DEFAULT_SPEED)

        total_tt += L / max(v_kmh / 3.6, 0.1)

        ln = lanes.get(eid, 1)
        if min_lanes_val is None or ln < min_lanes_val:
            min_lanes_val = ln

    eff_speed_kmh = (total_length / max(total_tt, 1e-9)) * 3.6

    return (
        total_length,
        eff_speed_kmh if has_speed else None,
        min_lanes_val if min_lanes_val is not None else 1,
        first_hw,
    )


def _store(new_eid, u, v, chain_eids,
           new_edges, new_lengths, new_maxspeed, new_lanes, new_highway,
           street_to_links, lengths, maxspeed, lanes, highway):
    """Compute merged attributes and append to output dicts."""
    m_len, m_spd, m_lanes, m_hw = _merge_attrs(
        chain_eids, lengths, maxspeed, lanes, highway
    )
    new_edges.append((u, v, new_eid))
    new_lengths[new_eid]     = m_len
    new_maxspeed[new_eid]    = m_spd
    new_lanes[new_eid]       = m_lanes
    new_highway[new_eid]     = m_hw
    street_to_links[new_eid] = chain_eids
