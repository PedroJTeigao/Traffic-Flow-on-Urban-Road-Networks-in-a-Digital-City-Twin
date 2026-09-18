"""
Analyse the per-link CFL time step distribution for the Gothenburg network.

For each link e, the CFL limit is:
    dt_e = L_e / max(v_e, w_e)

The global dt used in simulation is:
    dt = 0.9 * min(dt_e)

This script plots the distribution of dt_e values and identifies which
links are the most constraining.
"""
import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import dtcc_core as dtcc

# ── same bounds ───────────────────────────────────────────────────────────────
X0, Y0 = 319995.962899, 6399009.716755
B_DESO  = dtcc.Bounds(X0-1000, Y0-1000, X0+1000, Y0+1000)

W = 3.6   # backward wave speed (m/s)

# ── road data ─────────────────────────────────────────────────────────────────
print("Downloading road data …")
roads_model = dtcc.datasets.roads(bounds=B_DESO)
ra  = roads_model.to_arrays(include_attributes=True)

V0  = np.asarray(ra["vertices"], float)
E0  = np.asarray(ra["edges"]).reshape(-1, 2)
L0  = np.asarray(ra["lengths"], float)
at  = ra.get("attributes", {})
hw0 = at.get("highway")
ln0 = at.get("lanes")

NON_DRIVABLE = {"footway","steps","pedestrian","path","cycleway","bridleway",
                "track","construction","platform","service","corridor"}
keep = (np.array([str(h) not in NON_DRIVABLE for h in hw0], bool)
        if hw0 is not None else np.ones(len(E0), bool))

L_f = L0[keep]
HW  = [str(hw0[i]) for i in range(len(E0)) if keep[i]] if hw0 is not None else [""] * int(keep.sum())

SPEED_MS = {"motorway": 110/3.6, "trunk": 90/3.6, "primary": 60/3.6,
            "secondary": 50/3.6, "tertiary": 40/3.6,
            "residential": 30/3.6, "unclassified": 30/3.6}
speeds = np.array([SPEED_MS.get(hw, 40/3.6) for hw in HW])

# ── per-link CFL limit ────────────────────────────────────────────────────────
dt_links = L_f / np.maximum(speeds, W)   # dt_e = L_e / max(v_e, w_e)
dt_sim   = 0.9 * dt_links.min()

print(f"\nLinks:         {len(L_f)}")
print(f"dt_e  min:     {dt_links.min():.4f} s  →  dt_sim = {dt_sim:.4f} s")
print(f"dt_e  median:  {np.median(dt_links):.4f} s")
print(f"dt_e  mean:    {dt_links.mean():.4f} s")
print(f"dt_e  max:     {dt_links.max():.4f} s")
print(f"\nSimulation steps for 1 h: {int(3600 / dt_sim)}")

# most constraining links
idx_sorted = np.argsort(dt_links)
print("\nTop 10 most constraining links (smallest dt_e):")
print(f"{'eid':>6}  {'hw class':20}  {'L (m)':>8}  {'v (m/s)':>8}  {'dt_e (s)':>10}")
for i in idx_sorted[:10]:
    print(f"{i:>6}  {HW[i]:20}  {L_f[i]:>8.1f}  {speeds[i]:>8.2f}  {dt_links[i]:>10.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# top-left — full histogram
ax = axes[0, 0]
ax.hist(dt_links, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(dt_sim, color="red", linewidth=1.5, label=f"dt_sim = {dt_sim:.3f} s")
ax.axvline(np.median(dt_links), color="orange", linewidth=1.5,
           linestyle="--", label=f"median = {np.median(dt_links):.2f} s")
ax.set_xlabel("Per-link CFL limit  dt_e  (s)")
ax.set_ylabel("Number of links")
ax.set_title("Full distribution of CFL time step")
ax.legend()

# top-right — zoom on the smallest 5% (the constraining tail)
ax = axes[0, 1]
p5 = np.percentile(dt_links, 5)
tail = dt_links[dt_links <= p5]
ax.hist(tail, bins=60, color="tomato", edgecolor="white", linewidth=0.3)
ax.axvline(dt_sim, color="red", linewidth=1.5, label=f"dt_sim = {dt_sim:.3f} s")
ax.set_xlabel("Per-link CFL limit  dt_e  (s)")
ax.set_ylabel("Number of links")
ax.set_title(f"Zoom — smallest 5%  (≤ {p5:.2f} s,  {len(tail)} links)")
ax.legend()

# bottom-left — log scale full distribution
ax = axes[1, 0]
ax.hist(dt_links, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
ax.axvline(dt_sim, color="red", linewidth=1.5, label=f"dt_sim = {dt_sim:.3f} s")
ax.set_xscale("log")
ax.set_xlabel("Per-link CFL limit  dt_e  (s)  [log scale]")
ax.set_ylabel("Number of links")
ax.set_title("Log-scale distribution")
ax.legend()

# bottom-right — dt_e vs link length, coloured by highway class
ax = axes[1, 1]
classes = sorted(set(HW))
cmap = plt.get_cmap("tab10")
for k, cls in enumerate(classes):
    mask = np.array([h == cls for h in HW])
    ax.scatter(L_f[mask], dt_links[mask], s=6, alpha=0.5,
               color=cmap(k % 10), label=cls)
ax.set_xlabel("Link length  L_e  (m)")
ax.set_ylabel("CFL limit  dt_e  (s)")
ax.set_title("CFL limit vs link length by road class")
ax.set_xlim(0,1.5)
ax.set_ylim(0,0.2)
ax.legend(fontsize=7, markerscale=2)

plt.tight_layout()
OUT_DIR = os.path.join(REPO_ROOT, 'output')
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "dt_distribution.png")
plt.savefig(out, dpi=150)
print(f"\nSaved → {out}")
