# ECMI Modelling Week 2026 — Traffic Flow on a Digital City Twin (Part I: CTM)

This repository contains **Part I** of a two-part traffic model built during
ECMI Modelling Week 2026, Project 7 ("Traffic Flow on Digital City Twin"): a
**Cell Transmission Model (CTM)** simulating vehicle flow on the real road
network of a **2 km × 2 km study area in central Gothenburg, Sweden**
(EPSG:3006, SWEREF99 TM), built from OpenStreetMap road geometry and
Statistics Sweden (DeSO) demand data served through
[DTCC Core](https://github.com/dtcc-platform/dtcc-core), the Digital Twin
Cities Centre's platform.

The CTM discretises each OSM road segment (or, in `mode='roads'`, each street
between real intersections) as one cell with a triangular fundamental diagram,
and resolves flow through junctions using the Lebacque & Khoshyaran (2002)
node model. Demand is generated from an origin–destination gravity model
calibrated on DeSO population/employment/car-ownership statistics.

**Report reference:** cite this repository as the code release accompanying
the ECMI Modelling Week 2026, Project 7 group report, "Traffic Flow on a
Digital City Twin" — see [`repo URL`] and the Code Availability line at the
bottom of this file.

Part II (the LWR/Godunov PDE continuum model) is being developed separately
and will be added under [`pde/`](pde/README.md).

---

## 1. Repository layout

```
ecmi-mw2026-traffic/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  ctm/          # Part I — the CTM engine (formerly "traffic_engine_3")
  scripts/      # runnable experiments (see table below)
  pde/          # placeholder for Part II
```

`ctm/` is the `traffic_engine_3` package with its modules kept intact
(`cell.py`, `junction.py`, `network.py`, `utils.py`, `merge.py`,
`od_matrix.py`) — only the folder was renamed for the public release, and
the scripts' import lines and `sys.path` setup were updated to match
(`from ctm.utils import ...` instead of `from traffic_engine_3.utils import
...`, resolving `ctm/` from the repo root rather than assuming it sits next
to the script). No module was moved or split, and no model code was edited.

## 2. Install

```bash
git clone <repo URL>
cd ecmi-mw2026-traffic
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this code was tested against
(Python 3.12.10, Windows). See the note in that file about `dtcc-core` — the
installed build here is a development snapshot (`0.9.8.dev0`); if
`pip install dtcc-core` resolves to something with a different API on PyPI,
install from source instead:

```bash
pip install git+https://github.com/dtcc-platform/dtcc-core.git
```

### Data setup

No manual download is required. Every script in `scripts/` fetches its own
data at runtime through DTCC Core, for a fixed 2 km × 2 km bounding box
centred at `(X0, Y0) = (319995.96, 6399009.72)` in EPSG:3006:

```python
import dtcc_core as dtcc
B_DESO = dtcc.Bounds(X0-1000, Y0-1000, X0+1000, Y0+1000)

roads_model = dtcc.datasets.roads(bounds=B_DESO)          # OSM road graph
deso_obj    = dtcc.datasets.deso(bounds=B_DESO,             # DeSO statistics
                                  statistics=["population", "cars", "employment"])
```

`dtcc.datasets.roads()` pulls OpenStreetMap way/node data for the bounding
box and returns it as a graph (vertices, edges, per-edge `highway`, `lanes`,
`maxspeed` attributes). `dtcc.datasets.deso()` pulls Statistics Sweden's DeSO
(Demographic Statistical Areas) zone statistics — population, car ownership,
employment — for the same box, with zone centroids used as OD gravity-model
sources/sinks. Both are cached by DTCC Core on first fetch; if the network
call fails or times out, every script falls back to a small synthetic 6-node
grid so it can still run (clearly logged as `[WARNING] Gothenburg data
unavailable`).

## 3. Scripts

All scripts are run directly, e.g. `python scripts/test_gothenburg_compare_v4.py`
from the repo root (they add the repo root to `sys.path` themselves). Outputs
are written to `./output/`, which is created on demand and is git-ignored.

| Script | Experiment | Figures produced | Demand profile | Approx. cost |
|---|---|---|---|---|
| `scripts/test_gothenburg_compare.py` | OD variant 1 (baseline): `mode='links'` vs `mode='roads'`, default OD thresholds | `gothenburg_delay_compare.png`, `gothenburg_links.gif`, `gothenburg_roads.gif` | `build_network_od` defaults: `demand_scale=1e-5`, `high_prod/attr_threshold=0.05` (top 5% of nodes inject/absorb) | 120 min simulated, 2 networks built and stepped |
| `scripts/test_gothenburg_compare_v2.py` | OD variant 2: same links-vs-roads comparison, wider injection threshold (top 20% of nodes) | `gothenburg_avg_speed_v2.png`, `gothenburg_queue_length_v2.png`, `gothenburg_load_factor_v2.png`, `gothenburg_total_delay_v2.png`, `gothenburg_links_v2.gif`, `gothenburg_roads_v2.gif` | `high_prod_threshold=0.20`, `high_attr_threshold=0.20`, `demand_scale=1e-5` | 120 min simulated, 2 networks |
| `scripts/test_gothenburg_compare_v3.py` | OD variant 3: `mode='links'` only, **all** interior junctions inject+absorb, 10× production | `gothenburg_avg_speed_v3.png`, `_queue_length_v3.png`, `_load_factor_v3.png`, `_total_delay_v3.png`, `gothenburg_links_v3.gif` | `prod_scale=1e-4`, `attr_scale=1e-5`, `high_prod/attr_threshold=1.0` (every junction) | 120 min simulated, 1 network |
| `scripts/test_gothenburg_compare_v3_roads.py` | Same as v3, `mode='roads'` (merged streets) | `..._v3_roads.png` (×4), `gothenburg_roads_v3.gif` | same as v3 | 120 min simulated, 1 network |
| `scripts/test_gothenburg_compare_v4.py` | **The v4 comparison run**: `mode='links'`, dynamic rush-hour demand profile | `gothenburg_avg_speed_v4.png`, `_queue_length_v4.png`, `_load_factor_v4.png`, `_total_delay_v4.png`, `gothenburg_links_v4.gif` | Time-varying `prod_scale`: ×10 (0–20 min) → ×50 (20–60 min, morning peak) → ×100 (60–80 min, peak max) → ×10 (80–120 min, return); `attr_scale=1e-5` constant | 120 min simulated, 1 network |
| `scripts/test_gothenburg_compare_v4_roads.py` | v4 companion run, `mode='roads'` | `..._v4_roads.png` (×4), `gothenburg_roads_v4.gif` | same dynamic profile as v4 | 120 min simulated, 1 network |
| `scripts/test_gothenburg_compare_v5.py` | **The v5 clearance-phase run**: build-up → sustained congestion → clearance → empty | `gothenburg_avg_speed_v5.png`, `_queue_length_v5.png`, `_load_factor_v5.png`, `_total_delay_v5.png`, `gothenburg_links_v5.gif` | `(prod, attr)`: (1e-3, 1e-6) 0–30 min (build-up) and 30–60 min (sustained) → (1e-5, 1e-4) 60–90 min (clearance) and 90–120 min (empty) | 120 min simulated, 1 network |
| `scripts/analyse_dt.py` | Diagnostic (no CTM run): distribution of the per-link CFL limit `dt_e = L_e / max(v_e, w_e)` over the raw road graph | `dt_distribution.png` | n/a — pure network-geometry diagnostic, no demand | seconds; substantiates the CFL finding below |

**"Approx. cost"** is deliberately not given in wall-clock minutes: every
`compare*`/`v3`/`v4`/`v5` script simulates 7200 s (120 min) of network time,
and the global CFL step (see Known issues below) collapses to roughly 0.03 s,
i.e. **~240,000 time steps per network per run**. Wall-clock time then
depends heavily on network size and hardware, so we did not benchmark it —
budget it as a "leave it running" job, not a quick check. This repo's smoke
test (below) only verifies data loading and network construction, not a full
run.

### Scripts left out (and why)

The original working directory had ~19 exploratory `test_gothenburg_*.py`
scripts. Only the eight above are included. Left out:

| Script(s) | Reason |
|---|---|
| `test_gothenburg.py`, `test_gothenburg_v2.py` | Earliest prototypes: constant boundary inflow / an ad hoc DeSO-gateway hack, both superseded by the OD-gravity pipeline in `ctm/utils.py`'s `build_network_od`. They also import the **older, unshipped** `traffic_engine` package, not `traffic_engine_3`/`ctm`. |
| `test_gothenburg_te2.py` | Uses `traffic_engine_2`, a different (multi-cell-per-link, `M`-cell) engine variant. Not part of Part I / `ctm`, and not shipped here. |
| `test_gothenburg_absorption.py` | Exploratory diagnostic of the injected/absorbed vehicle balance; imports `traffic_engine_3` correctly but also patches dead-end and boundary (80 m margin) nodes to `SinkNode` by hand — a pre-`build_network_od` pattern superseded by the native `InjectionAbsorptionJunction` handling exercised in v3–v5. |
| `test_gothenburg_od_dist.py`, `_main_roads`, `_priority`, `test_gothenburg_od_time.py`, `_main_roads`, `_priority` (6 scripts) | All import `from traffic_engine.utils import ...` / `from traffic_engine.od_matrix import ...` — the **older, unshipped** `traffic_engine` package, not `ctm`. Including them would give a repo with import errors. Their OD-gravity logic (`build_od_dist`/`build_od_time`) was carried forward into `ctm/od_matrix.py` and is exercised through `build_network_od` by every included `compare*` script. |
| `mini_test.py`, `explore_roads.py` | Ad hoc, interactive data-exploration snippets (plot the raw DTCC road graph; one filters by street name). No `traffic_engine`/`ctm` involvement, not an "experiment". |

If you want any of these anyway (e.g. to compare `traffic_engine` vs `ctm`
behaviour), they still exist in the original project folder outside this
repository — they were not deleted, only left out of this release.

## 4. Smoke test

Per the packaging brief, the full 7200 s simulations were **not** re-run
during packaging. Instead, a data-loading-only smoke test was used to verify
that `dtcc_core` fetches the Gothenburg bounding box, `ctm.utils.from_dtcc`
and `ctm.utils.build_network` construct a `Network` from it, and one `net.step()`
succeeds:

```bash
python -c "
import dtcc_core as dtcc
from ctm.utils import from_dtcc, build_network
X0, Y0 = 319995.962899, 6399009.716755
b = dtcc.Bounds(X0-1000, Y0-1000, X0+1000, Y0+1000)
ra = dtcc.datasets.roads(bounds=b).to_arrays(include_attributes=True)
print('roads OK:', len(ra[\"vertices\"]), 'vertices,', len(ra[\"edges\"]), 'edges')
"
```

**Result: PASSED** on this machine (2026-09-18) — `ctm` package imports
resolved cleanly; `dtcc.datasets.roads()` returned 19,119 vertices / 21,565
edges for the bounding box; `build_network` constructed a `Network` from it
(`dt ≈ 0.0016 s` — smaller than the ~0.03 s quoted in Known Issues below
because this quick check does not apply the `NON_DRIVABLE` highway-class
filter the real scripts use, so it also counts footways/paths with very
short segments); one `net.step()` executed without error; `dtcc.datasets.deso()`
returned 30 DeSO zones. The full 7200 s simulations were not run.

## 5. Known issues

These were found during a fact-check of the model against the report's
figures. They are **not fixed here** — the brief for this release was to
document, not correct, so the shipped code still reproduces the report's
numbers exactly.

1. **The jam density is ~10× a realistic value.**
   `ctm/cell.py:24` sets `K = 1500 * lanes / 1000` veh/m, i.e. **1500 veh/km
   per lane** (1.5 veh/m). A realistic urban jam density is closer to
   130–150 veh/km per lane. Storage capacity `N = K * L`
   (`ctm/cell.py:25`) is inflated by the same factor — for this network it
   sums to roughly **136,000 vehicles** of total storage — which means
   **every absolute queue length, load factor, and delay number reported by
   these scripts is inflated accordingly.** Relative comparisons (links vs.
   roads, peak vs. off-peak) are still meaningful; absolute magnitudes are
   not calibrated to reality.

2. **The triangular fundamental diagram is internally inconsistent as a
   consequence of (1).** The demand branch `D = min(C, v·ρ)`
   (`ctm/cell.py:32-37`) and supply branch `S = min(C, w·(K-ρ))`
   (`ctm/cell.py:39-44`) are meant to meet at the critical density
   `ρ_c = C/v`. With `K` inflated ~10×, `w·(K - ρ_c) ≈ 10·C` — the supply
   branch only reaches capacity `C` at a density far below `K`, so the two
   branches do not intersect at `ρ_c` as a proper triangular FD requires.
   The congested-speed branch computed in `ctm/network.py:71-79`
   (`speed = w·(K-ρ)/ρ` for `ρ > ρ_c`) is therefore discontinuous at
   `ρ_c`, jumping from free-flow speed to a much higher value before
   decaying — the classic triangular-FD wave-speed relation
   `w = C/(K-ρ_c)` is not satisfied by the fixed `w = 3.6` m/s constant
   (`ctm/cell.py:18`, `ctm/utils.py` default) together with the inflated `K`.

3. **The OSM `maxspeed` tag is never actually read.** Every script that
   loads live Gothenburg data builds `MS = [None] * len(E_f)` before
   calling `from_dtcc` — e.g. `scripts/test_gothenburg_compare_v4.py:101`
   (and the equivalent line in every other `compare*` script; see the grep
   pattern `MS = [None]` if you want the rest). `ctm/utils.py:65-69`'s
   `build_network` *would* use `road_data['maxspeed'][eid]` if it were
   populated, but since it is always `None`, every link's free-flow speed
   comes from the per-highway-class defaults in `SPEED_DEFAULTS`
   (`ctm/utils.py:18-26`), never from the OSM tag itself, even where OSM
   has one.

4. **The global CFL step is ~0.03 s because of sub-metre edges.**
   `ctm/utils.py:74-77` sets `dt = min_e(L_e / max(v_e, w_e)) * safety`
   (`safety=0.9` by default) — a single edge with `L_e` of a few tenths of
   a metre forces this global minimum down to roughly **0.030 s**, giving
   **~240,000 steps** for a 7200 s (2 h) run. `scripts/analyse_dt.py`
   reproduces this distribution directly from the raw DTCC road graph (no
   `ctm` import needed) and prints/plots the most constraining links.

5. **The 80 m boundary-margin sink rule is not part of the v4/v5 pipeline.**
   An earlier prototype (`test_gothenburg_absorption.py`, and the six
   `test_gothenburg_od_dist*`/`test_gothenburg_od_time*` scripts — see
   "Scripts left out" above; none of these are included in this repository)
   forced every node within 80 m of the bounding-box edge to a `SinkNode`
   (`BORDER_MARGIN = 80.0`, applied as a post-`build_network` patch). This
   rule was dropped when the pipeline moved to `build_network_od` /
   `InjectionAbsorptionJunction` for v3–v5 (`ctm/utils.py`'s
   `build_network_od`, used by every included `compare*` script) and does
   **not** affect the report's v4/v5 figures — boundary nodes there are
   plain `SourceNode`/`SinkNode`/`Junction` by degree only, exactly as
   `build_network` (`ctm/utils.py:44-110`) assigns them.

## 6. Credits

Built during **ECMI Modelling Week 2026**, hosted by Lund University. The
problem was posed by the **Digital Twin Cities Centre (DTCC)** at Chalmers
University of Technology. Road network data from **OpenStreetMap**
contributors (© OpenStreetMap contributors, [ODbL-licensed](https://opendatacommons.org/licenses/odbl/) —
any redistribution of the underlying map data must comply with the ODbL);
demographic and employment statistics from **Statistics Sweden (SCB)**'s
DeSO (Demografiska statistikområden) dataset. Data access via
[DTCC Core](https://github.com/dtcc-platform/dtcc-core).

Authors: ECMI Modelling Week 2026, Project 7 group.

## Code availability

The CTM implementation (Part I) used to produce the traffic-flow results in
this report is openly available at [repo URL]; the LWR/Godunov PDE model
(Part II) will be added to the same repository under `pde/`.
