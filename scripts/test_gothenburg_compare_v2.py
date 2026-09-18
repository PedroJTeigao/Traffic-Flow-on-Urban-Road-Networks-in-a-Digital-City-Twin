"""
Compare CTM simulation: mode='links' vs mode='roads' on Gothenburg network.
v2: high_prod_threshold=0.20 / high_attr_threshold=0.20 (top 20% → more
    InjectionJunctions, more mid-network vehicle injection).

Simulates 120 minutes (7200 s) with 60 snapshots × 2 min each.

Outputs (all in ./output/):
  gothenburg_delay_compare_v2.png  — total network delay over time, both modes
  gothenburg_links_v2.gif          — animated VCR map, one cell per OSM link
  gothenburg_roads_v2.gif          — animated VCR map, one cell per merged street
"""
import sys
import os
import io

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from PIL import Image

from ctm.utils import from_dtcc, build_network_od

OUT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
GOTHENBURG_OK = False
try:
    import dtcc_core as dtcc

    X0, Y0 = 319995.962899, 6399009.716755
    B_DESO  = dtcc.Bounds(X0-1000, Y0-1000, X0+1000, Y0+1000)

    print("Downloading road data …")
    roads_model = dtcc.datasets.roads(bounds=B_DESO)
    ra  = roads_model.to_arrays(include_attributes=True)

    V0  = np.asarray(ra["vertices"], float)
    E0  = np.asarray(ra["edges"]).reshape(-1, 2)
    L0  = np.asarray(ra["lengths"], float)
    at  = ra.get("attributes", {})
    hw0 = at.get("highway")
    ln0 = at.get("lanes")

    NON_DRIVABLE = {
        "footway", "steps", "pedestrian", "path", "cycleway", "bridleway",
        "track", "construction", "platform", "service", "corridor",
    }
    keep = (
        np.array([str(h) not in NON_DRIVABLE for h in hw0], bool)
        if hw0 is not None else np.ones(len(E0), bool)
    )

    E_f = E0[keep]
    L_f = L0[keep]
    HW  = (
        [str(hw0[i]) for i in range(len(E0)) if keep[i]]
        if hw0 is not None else [""] * int(keep.sum())
    )
    LAN = []
    for i in range(len(E0)):
        if not keep[i]:
            continue
        raw = ln0[i] if ln0 is not None else None
        try:
            val = max(int(float(str(raw).split(";")[0].split()[0])), 1)
        except Exception:
            val = 1
        LAN.append(val)
    MS = [None] * len(E_f)

    used  = np.unique(E_f.reshape(-1))
    remap = -np.ones(len(V0), int)
    remap[used] = np.arange(len(used))
    V   = V0[used, :2]
    E_r = remap[E_f]

    print("Downloading DeSO data …")
    deso_obj = dtcc.datasets.deso(
        bounds=B_DESO, statistics=["population", "cars", "employment"]
    )
    da       = deso_obj.to_arrays()
    cen      = np.asarray(da["centroids"])
    z_cars   = np.asarray(da["fields"]["cars_in_traffic"],          float).ravel()
    z_employ = np.asarray(da["fields"]["employed_residents_total"], float).ravel()

    road_data = from_dtcc(edges=E_r, lengths=L_f, maxspeed=MS, lanes=LAN, highway=HW)
    print(f"DeSO: {len(cen)} zones | cars: {z_cars.sum():.0f} | "
          f"employment: {z_employ.sum():.0f}")
    GOTHENBURG_OK = True

except Exception as exc:
    print(f"[WARNING] Gothenburg data unavailable ({exc}). Using synthetic 6-node grid.")

if not GOTHENBURG_OK:
    V = np.array([
        [0., 0.], [1000., 0.], [2000., 0.],
        [0., 1000.], [1000., 1000.], [2000., 1000.],
    ], float)
    E_r = np.array([
        [0, 1], [1, 2], [3, 4], [4, 5],
        [0, 3], [1, 4], [2, 5],
    ], int)
    L_f = np.full(len(E_r), 1000., float)
    HW  = ['residential'] * len(E_r)
    LAN = [1] * len(E_r)
    MS  = [None] * len(E_r)
    cen      = np.array([[500., 500.], [1500., 500.]], float)
    z_cars   = np.array([100., 50.], float)
    z_employ = np.array([50., 100.], float)
    road_data = from_dtcc(edges=E_r, lengths=L_f, maxspeed=MS, lanes=LAN, highway=HW)

print(f"\nOriginal: {len(V)} nodes, {len(road_data['edges'])} links")

# ── 2. Build both networks (top 20% threshold) ────────────────────────────────
print("\n[1/2] Building mode=links network …")
net_A = build_network_od(
    road_data, V, cen, z_cars, z_employ,
    mode='links', T_max=180, demand_scale=1e-5,
    high_prod_threshold=0.20,
    high_attr_threshold=0.20,
)

print("\n[2/2] Building mode=roads network …")
net_B = build_network_od(
    road_data, V, cen, z_cars, z_employ,
    mode='roads', T_max=180, demand_scale=1e-5,
    high_prod_threshold=0.20,
    high_attr_threshold=0.20,
)

n_links   = len(net_A.links)
n_streets = len(net_B.links)
reduction = 100.0 * (1 - n_streets / n_links) if n_links > 0 else 0.0
print(f"\nMode links : {n_links} links")
print(f"Mode roads : {n_streets} streets  ({reduction:.1f}% reduction)")

# ── 3. Segment geometry ───────────────────────────────────────────────────────
edge_ids_A = [eid for u, v, eid in net_A.edges]
segs_A     = [[V[u].tolist(), V[v].tolist()] for u, v, eid in net_A.edges]

jp         = net_B.junction_positions
edge_ids_B = [eid for u, v, eid in net_B.edges]
segs_B     = [
    [list(jp.get(u, V[u])), list(jp.get(v, V[v]))]
    for u, v, eid in net_B.edges
]
lengths_B  = [net_B.links[eid].L for eid in edge_ids_B]
max_L_B    = max(lengths_B) if lengths_B else 1.0
lws_B      = [1.0 + 3.0 * (L / max_L_B) for L in lengths_B]

# ── 4. Simulate 120 min, 60 snapshots × 2 min ────────────────────────────────
T_SIM    = 7200.0
N_FRAMES = 60
DT_SNAP  = T_SIM / N_FRAMES   # 120 s per snapshot

snapshots_A = []
snapshots_B = []

print(f"\nSimulating {T_SIM/60:.0f} min — {N_FRAMES} snapshots × {DT_SNAP:.0f} s ({DT_SNAP/60:.0f} min each) …")
for step in range(N_FRAMES):
    net_A.run(DT_SNAP)
    net_B.run(DT_SNAP)
    m_A = net_A.metrics()
    m_B = net_B.metrics()
    snapshots_A.append((net_A.t, m_A))
    snapshots_B.append((net_B.t, m_B))
    if (step + 1) % 10 == 0:
        print(f"  snapshot {step+1:2d}/{N_FRAMES}  t = {net_A.t/60:.0f} min")

# ── 5. Metric helpers ────────────────────────────────────────────────────────
def total_delay(net, m):
    """Total network delay in veh·s: Σ_e max(0, tt_e - ff_tt_e) * n_e."""
    td = 0.0
    for eid, link in net.links.items():
        ff_tt = link.L / link.v if link.v > 0 else 0.0
        delay = max(0.0, m[eid]['travel_time'] - ff_tt)
        td += delay * link.n
    return td

def avg_speed(net, m):
    """Length-weighted mean free-flow speed across all links (m/s)."""
    total_L = sum(link.L for link in net.links.values())
    if total_L == 0:
        return 0.0
    return sum(m[eid]['speed'] * link.L for eid, link in net.links.items()) / total_L

def queue_length(net, m):
    """Total vehicles sitting on congested links (rho > rho_critical) (veh)."""
    q = 0.0
    for eid, link in net.links.items():
        rho_c = link.C / link.v if link.v > 0 else link.K
        if m[eid]['rho'] > rho_c:
            q += link.n
    return q

def load_factor(net, m):
    """Mean load factor across all links: mean(n_e / N_e)."""
    vals = [link.n / link.N for link in net.links.values() if link.N > 0]
    return sum(vals) / len(vals) if vals else 0.0

times_A   = [t for t, _ in snapshots_A]
times_B   = [t for t, _ in snapshots_B]
delays_A  = [total_delay(net_A, m)  for _, m in snapshots_A]
delays_B  = [total_delay(net_B, m)  for _, m in snapshots_B]
speeds_A  = [avg_speed(net_A, m)    for _, m in snapshots_A]
speeds_B  = [avg_speed(net_B, m)    for _, m in snapshots_B]
queues_A  = [queue_length(net_A, m) for _, m in snapshots_A]
queues_B  = [queue_length(net_B, m) for _, m in snapshots_B]
loads_A   = [load_factor(net_A, m)  for _, m in snapshots_A]
loads_B   = [load_factor(net_B, m)  for _, m in snapshots_B]

# ── 6. Four metric plots (links subplot | roads subplot) ──────────────────────
METRICS = [
    ('Average Speed',        'avg_speed',      speeds_A,  speeds_B,  'm/s',    'gothenburg_avg_speed_v2.png'),
    ('Queue Length',         'queue_length',   queues_A,  queues_B,  'veh',    'gothenburg_queue_length_v2.png'),
    ('Load Factor (ρ/K)',    'load_factor',    loads_A,   loads_B,   '—',      'gothenburg_load_factor_v2.png'),
    ('Total Network Delay',  'total_delay',    delays_A,  delays_B,  'veh·s',  'gothenburg_total_delay_v2.png'),
]

for title, _, vals_A, vals_B, ylabel, fname in METRICS:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, vals, label in [(ax_l, vals_A, 'mode=links'), (ax_r, vals_B, 'mode=roads')]:
        ax.plot(times_A, vals, linewidth=2, color='steelblue' if 'links' in label else 'darkorange')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{title} — {label}')
        ax.grid(True, alpha=0.3)
    ax_l.set_ylabel(f'{title} ({ylabel})')
    fig.suptitle(f'Gothenburg CTM v2 — {title}  (threshold=0.20)', fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved → output/{fname}")

print(f"\nSaved → output/gothenburg_delay_compare_v2.png (included above as total_delay)")

# ── 7. GIF helpers ────────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=0.0, vmax=1.5)


def _frame_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _save_gif(frames, path, fps=4):
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )


# ── 8. GIF — mode=links ───────────────────────────────────────────────────────
print("Rendering gothenburg_links_v2.gif …")
frames_A = []
for t, m in snapshots_A:
    vcr = np.array([m[eid]['vcr'] for eid in edge_ids_A])
    fig, ax = plt.subplots(figsize=(10, 9))
    lc = LineCollection(segs_A, array=vcr, cmap=cmap, norm=norm, linewidths=1.5)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label='VCR')
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'mode=links  (thr=0.20)   t = {t:.0f}s')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.tight_layout()
    frames_A.append(_frame_to_pil(fig))

gif_A = os.path.join(OUT_DIR, 'gothenburg_links_v2.gif')
_save_gif(frames_A, gif_A, fps=4)
print("Saved → output/gothenburg_links_v2.gif")

# ── 9. GIF — mode=roads ───────────────────────────────────────────────────────
print("Rendering gothenburg_roads_v2.gif …")
frames_B = []
for t, m in snapshots_B:
    vcr = np.array([m[eid]['vcr'] for eid in edge_ids_B])
    fig, ax = plt.subplots(figsize=(10, 9))
    lc = LineCollection(segs_B, array=vcr, cmap=cmap, norm=norm, linewidths=lws_B)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label='VCR')
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'mode=roads  (thr=0.20)   t = {t:.0f}s')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.tight_layout()
    frames_B.append(_frame_to_pil(fig))

gif_B = os.path.join(OUT_DIR, 'gothenburg_roads_v2.gif')
_save_gif(frames_B, gif_B, fps=4)
print("Saved → output/gothenburg_roads_v2.gif")

# ── 10. Final stats ───────────────────────────────────────────────────────────
total_len_A   = sum(link.L for link in net_A.links.values()) / 1000.0
total_len_B   = sum(link.L for link in net_B.links.values()) / 1000.0
len_reduction = 100.0 * (1 - total_len_B / total_len_A) if total_len_A > 0 else 0.0
final_m_A     = snapshots_A[-1][1]
final_m_B     = snapshots_B[-1][1]

print(f"\n{'='*68}")
print(f"  FINAL STATS at t = {T_SIM/60:.0f} min  [threshold=0.20]")
print(f"{'='*68}")
print(f"\n  Network stats:")
print(f"    mode=links : {n_links:5d} links,   total length {total_len_A:.1f} km")
print(f"    mode=roads : {n_streets:5d} streets, total length {total_len_B:.1f} km")
print(f"    Reduction  : {reduction:.1f}%  (links→streets) | "
      f"{len_reduction:.1f}% (length)")

print(f"\n  Delay stats:")
print(f"    mode=links total delay : {delays_A[-1]:.1f} veh·s")
print(f"    mode=roads total delay : {delays_B[-1]:.1f} veh·s")

def _top10(net, m, label):
    rows = sorted(
        [(eid, link.L, m[eid]['vcr'], m[eid]['speed'], m[eid]['travel_time'])
         for eid, link in net.links.items()],
        key=lambda r: r[2], reverse=True
    )[:10]
    print(f"\n  Top 10 congested — {label}  (vcr desc):")
    print(f"    {'id':>6}  {'length(m)':>10}  {'vcr':>6}  {'speed(m/s)':>10}  {'delay(s)':>9}")
    print(f"    {'-'*6}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*9}")
    for eid, L, vcr, spd, tt in rows:
        link  = net.links[eid]
        ff_tt = link.L / link.v if link.v > 0 else 0.0
        delay = max(0.0, tt - ff_tt)
        print(f"    {eid:>6}  {L:>10.1f}  {vcr:>6.3f}  {spd:>10.3f}  {delay:>9.2f}")

_top10(net_A, final_m_A, 'mode=links')
_top10(net_B, final_m_B, 'mode=roads')
