"""
Gothenburg CTM — mode='links', rush hour + clearance.
v5: dynamic prod_scale and attr_scale — heavy injection during rush hour,
    then strong absorption to clear the network.

Demand profile:
  0  – 30 min  →  prod=1e-3  attr=1e-6   (build-up: inject fast, absorb slow)
  30 – 60 min  →  prod=1e-3  attr=1e-6   (sustained congestion)
  60 – 90 min  →  prod=1e-5  attr=1e-4   (clearance: inject slow, absorb fast)
  90 – 120 min →  prod=1e-5  attr=1e-4   (network empties)

All interior junctions inject+absorb (threshold=1.0).

Outputs (./output/):
  gothenburg_avg_speed_v5.png
  gothenburg_queue_length_v5.png
  gothenburg_load_factor_v5.png
  gothenburg_total_delay_v5.png
  gothenburg_links_v5.gif
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

from ctm.utils    import from_dtcc, build_network_od
from ctm.junction import SourceNode, InjectionJunction, InjectionAbsorptionJunction, AbsorptionJunction

OUT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── demand profile ────────────────────────────────────────────────────────────
PROFILE = [
    (0,        30 * 60,  1e-3,  1e-6),   # 0–30 min   build-up
    (30 * 60,  60 * 60,  1e-3,  1e-6),   # 30–60 min  sustained
    (60 * 60,  90 * 60,  1e-5,  1e-4),   # 60–90 min  clearance
    (90 * 60, 120 * 60,  1e-5,  1e-4),   # 90–120 min empty
]

def get_scales(t):
    """Return (prod_scale, attr_scale) for simulation time t (s)."""
    for t_start, t_end, ps, as_ in PROFILE:
        if t_start <= t < t_end:
            return ps, as_
    return 1e-5, 1e-4

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

# ── 2. Build network with base prod_scale (overridden dynamically) ────────────
print("\nBuilding network …")
net = build_network_od(
    road_data, V, cen, z_cars, z_employ,
    mode='links',
    T_max=180,
    prod_scale=1e-3,
    attr_scale=1e-6,
    high_prod_threshold=1.0,
    high_attr_threshold=1.0,
)
print(f"Links: {len(net.links)}")

# ── 3. Store base rates ───────────────────────────────────────────────────────
source_base  = {}
inject_base  = {}
absorb_base  = {}

for nid, nd in net.nodes.items():
    if isinstance(nd, SourceNode):
        source_base[nid] = nd.demand_fn(0.0)                     # at prod_scale=1e-3
    if isinstance(nd, (InjectionJunction, InjectionAbsorptionJunction)):
        inject_base[nid] = dict(nd.extra_inflow)                  # at prod_scale=1e-3
    if isinstance(nd, (AbsorptionJunction, InjectionAbsorptionJunction)):
        absorb_base[nid] = nd.absorption_rate                     # at attr_scale=1e-6

def apply_scales(t):
    """Update all injection and absorption rates for time t."""
    ps, as_ = get_scales(t)
    prod_ratio = ps  / 1e-3   # relative to base prod_scale
    attr_ratio = as_ / 1e-6   # relative to base attr_scale

    for nid, base in source_base.items():
        nd = net.nodes[nid]
        r  = base * prod_ratio
        nd.demand_fn = lambda tt, r=r: r

    for nid, base_inflow in inject_base.items():
        nd = net.nodes[nid]
        nd.extra_inflow = {fid: v * prod_ratio for fid, v in base_inflow.items()}

    for nid, base_abs in absorb_base.items():
        nd = net.nodes[nid]
        nd.absorption_rate = base_abs * attr_ratio

# ── 4. Segment geometry ───────────────────────────────────────────────────────
edge_ids = [eid for u, v, eid in net.edges]
segs     = [[V[u].tolist(), V[v].tolist()] for u, v, eid in net.edges]

# ── 5. Simulate 120 min, 60 snapshots × 2 min ────────────────────────────────
T_SIM    = 7200.0
N_FRAMES = 60
DT_SNAP  = T_SIM / N_FRAMES   # 120 s per snapshot

snapshots = []

print(f"\nSimulating {T_SIM/60:.0f} min — {N_FRAMES} snapshots × {DT_SNAP:.0f} s …")
for step in range(N_FRAMES):
    apply_scales(net.t)
    net.run(DT_SNAP)
    snapshots.append((net.t, net.metrics()))
    if (step + 1) % 10 == 0:
        ps, as_ = get_scales(net.t)
        print(f"  snapshot {step+1:2d}/{N_FRAMES}  t = {net.t/60:.0f} min  "
              f"prod={ps:.0e}  attr={as_:.0e}")

# ── 6. Metric helpers ─────────────────────────────────────────────────────────
def total_delay(net, m):
    td = 0.0
    for eid, link in net.links.items():
        ff_tt = link.L / link.v if link.v > 0 else 0.0
        td += max(0.0, m[eid]['travel_time'] - ff_tt) * link.n
    return td

def avg_speed(net, m):
    total_n = sum(link.n for link in net.links.values())
    if total_n == 0:
        total_L = sum(link.L for link in net.links.values())
        return sum(link.v * link.L for link in net.links.values()) / total_L if total_L > 0 else 0.0
    return sum(m[eid]['speed'] * link.n for eid, link in net.links.items()) / total_n

def queue_length(net, m):
    q = 0.0
    for eid, link in net.links.items():
        rho_c = link.C / link.v if link.v > 0 else link.K
        if m[eid]['rho'] > rho_c:
            q += link.n
    return q

def load_factor(net, m):
    vals = [link.n / link.N for link in net.links.values() if link.N > 0]
    return sum(vals) / len(vals) if vals else 0.0

times  = [t for t, _ in snapshots]
delays = [total_delay(net, m)  for _, m in snapshots]
speeds = [avg_speed(net, m)    for _, m in snapshots]
queues = [queue_length(net, m) for _, m in snapshots]
loads  = [load_factor(net, m)  for _, m in snapshots]

# transitions in seconds
TRANSITIONS = [30*60, 60*60, 90*60]
T_LABELS    = ['build-up→sustained', 'sustained→clearance', 'clearance→empty']
T_COLORS    = ['#e74c3c', '#e67e22', '#27ae60']

# ── 7. Metric plots ───────────────────────────────────────────────────────────
METRICS = [
    ('Average Speed',       speeds,  'm/s',   'gothenburg_avg_speed_v5.png'),
    ('Queue Length',        queues,  'veh',   'gothenburg_queue_length_v5.png'),
    ('Load Factor (ρ/K)',   loads,   '—',     'gothenburg_load_factor_v5.png'),
    ('Total Network Delay', delays,  'veh·s', 'gothenburg_total_delay_v5.png'),
]

for title, vals, ylabel, fname in METRICS:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, vals, linewidth=2, color='steelblue', zorder=3)

    for t_tr, lbl, col in zip(TRANSITIONS, T_LABELS, T_COLORS):
        ax.axvline(t_tr, color=col, linestyle='--', linewidth=1.4, zorder=2)
        ax.text(t_tr + 40, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                lbl, color=col, fontsize=7, va='top')

    ax.axvspan(0,       30*60, alpha=0.07, color='red',    label='Build-up (prod=1e-3, attr=1e-6)')
    ax.axvspan(30*60,   60*60, alpha=0.07, color='orange', label='Sustained (prod=1e-3, attr=1e-6)')
    ax.axvspan(60*60,   90*60, alpha=0.07, color='green',  label='Clearance (prod=1e-5, attr=1e-4)')
    ax.axvspan(90*60,   T_SIM, alpha=0.07, color='blue',   label='Empty (prod=1e-5, attr=1e-4)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{title} ({ylabel})')
    ax.set_title(f'Gothenburg CTM v5 — {title}\n'
                 f'mode=links | rush hour + clearance | all junctions')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)
    print(f"Saved → output/{fname}")

# ── 8. GIF ────────────────────────────────────────────────────────────────────
cmap = plt.cm.RdYlGn_r
norm = mcolors.Normalize(vmin=0.0, vmax=1.5)

def _frame_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

print("Rendering gothenburg_links_v5.gif …")
frames = []
for t, m in snapshots:
    vcr          = np.array([m[eid]['vcr'] for eid in edge_ids])
    ps, as_      = get_scales(t)
    if   t <  30*60: phase = 'Build-up'
    elif t <  60*60: phase = 'Sustained'
    elif t <  90*60: phase = 'Clearance'
    else:            phase = 'Empty'
    fig, ax = plt.subplots(figsize=(10, 9))
    lc = LineCollection(segs, array=vcr, cmap=cmap, norm=norm, linewidths=1.5)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label='VCR')
    ax.autoscale(); ax.set_aspect('equal')
    ax.set_title(f'mode=links | rush hour + clearance | all junctions\n'
                 f'{phase}  |  prod={ps:.0e}  attr={as_:.0e}  |  t = {t/60:.0f} min')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    fig.tight_layout()
    frames.append(_frame_to_pil(fig))

gif_path = os.path.join(OUT_DIR, 'gothenburg_links_v5.gif')
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=250, loop=0)
print("Saved → output/gothenburg_links_v5.gif")

# ── 9. Final stats ────────────────────────────────────────────────────────────
total_len = sum(link.L for link in net.links.values()) / 1000.0
final_m   = snapshots[-1][1]

print(f"\n{'='*68}")
print(f"  FINAL STATS at t = {T_SIM/60:.0f} min  [v5: rush hour + clearance]")
print(f"{'='*68}")
print(f"  Links: {len(net.links)} | total length: {total_len:.1f} km")
print(f"  Total delay  : {delays[-1]:.1f} veh·s")
print(f"  Avg speed    : {speeds[-1]:.3f} m/s")
print(f"  Queue length : {queues[-1]:.1f} veh")
print(f"  Load factor  : {loads[-1]:.4f}")

rows = sorted(
    [(eid, link.L, final_m[eid]['vcr'], final_m[eid]['speed'], final_m[eid]['travel_time'])
     for eid, link in net.links.items()],
    key=lambda r: r[2], reverse=True
)[:10]
print(f"\n  Top 10 congested (vcr desc):")
print(f"    {'id':>6}  {'length(m)':>10}  {'vcr':>6}  {'speed(m/s)':>10}  {'delay(s)':>9}")
print(f"    {'-'*6}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*9}")
for eid, L, vcr, spd, tt in rows:
    link  = net.links[eid]
    ff_tt = link.L / link.v if link.v > 0 else 0.0
    print(f"    {eid:>6}  {L:>10.1f}  {vcr:>6.3f}  {spd:>10.3f}  {max(0., tt-ff_tt):>9.2f}")
