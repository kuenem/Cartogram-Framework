import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# ----------------------------------------------------------------------
# Shared style (kept close to plot_poly.py's plain matplotlib look, just
# with a bit more polish since these are meant to go straight into LaTeX)
# ----------------------------------------------------------------------
COLOR_MAP = "#8A8F98"        # original / "map" geometry -> neutral gray
COLOR_CARTOGRAM = "#2E86AB"  # cartogram / transformed geometry -> blue
COLOR_TARGET = "#F26419"     # target / desired value -> orange
COLOR_GOOD = "#4C9A2A"       # matches / correct -> green
COLOR_BAD = "#D64550"        # errors / mismatches -> red
COLOR_GAP = "#D64550"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})


def _shoelace_area(points):
    """Local shoelace-area helper (mirrors polygon_areanp from src.geometry
    so this file has no dependency on the project's src package)."""
    points = np.asarray(points, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _new_ax(figsize=(6, 5), title=""):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(title, pad=14)
    return fig, ax


def _fill_poly(ax, points, color, alpha=0.35, label=None, lw=1.6, ls="-"):
    points = np.asarray(points, dtype=float)
    closed = np.vstack([points, points[0]])
    ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=alpha)
    ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=lw, linestyle=ls, label=label)


# ----------------------------------------------------------------------
# 1. Cartographic error (epsilon / xi) — Section 4.4.1
# ----------------------------------------------------------------------
def explain_cartographic_error():
    """Shows a region's original area A0, its target area A*, and the area
    the cartogram actually achieved A — the three quantities epsilon/xi
    are computed from."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    base = np.array([[0, 0], [2, 0], [2.4, 1.6], [1, 2.2], [-0.3, 1.3]])

    # (a) original region, area A0
    _fill_poly(axes[0], base, COLOR_MAP)
    c = base.mean(axis=0)
    axes[0].scatter(*c, color="black", s=15, zorder=5)
    axes[0].set_title(r"Map region, area $A_i^{(0)}$")
    axes[0].text(0.5, -0.06, r"$A_i^{(0)}$", ha="center", va="top", fontsize=12,
                 transform=axes[0].transAxes)

    # (b) target: scaled up (this is what A* asks for)
    target = (base - c) * 1.6 + c
    _fill_poly(axes[1], base, COLOR_MAP, alpha=0.12, ls="--")
    _fill_poly(axes[1], target, COLOR_TARGET, alpha=0.35)
    axes[1].set_title(r"Desired target area $A_i^{*}$")
    axes[1].text(0.5, -0.06, r"$A_i^{*}$ (from the data value)", ha="center", va="top",
                 fontsize=11, color=COLOR_TARGET, transform=axes[1].transAxes)

    # (c) achieved cartogram area: slightly under target -> visualizes error
    achieved = (base - c) * 1.35 + c
    _fill_poly(axes[2], target, COLOR_TARGET, alpha=0.12, ls="--")
    _fill_poly(axes[2], achieved, COLOR_CARTOGRAM, alpha=0.4)
    axes[2].set_title(r"Achieved cartogram area $A_i$")
    a_target = _shoelace_area(target)
    a_achieved = _shoelace_area(achieved)
    eps = abs(a_achieved - a_target) / max(a_achieved, a_target)
    axes[2].text(0.5, -0.06,
                 rf"$\epsilon_i = \dfrac{{|A_i-A_i^{{*}}|}}{{\max(A_i,A_i^{{*}})}} \approx {eps:.2f}$",
                 ha="center", va="top", fontsize=11, color=COLOR_BAD, transform=axes[2].transAxes)

    # fig.suptitle("Cartographic error: how far the achieved area misses the target",
    #               fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Shape deformation / Hamming distance — Section 4.4.2
# ----------------------------------------------------------------------
def _normalize_shape(points):
    """Centre on centroid and rescale to unit area, as required by the
    Hamming-distance definition (Eq. 4.3)."""
    points = np.asarray(points, dtype=float)
    c = points.mean(axis=0)
    centred = points - c
    area = _shoelace_area(centred)
    scale = 1.0 / np.sqrt(area) if area > 0 else 1.0
    return centred * scale


def explain_hamming_distance():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    map_poly = np.array([[0, 0], [2, 0.2], [2.3, 1.8], [0.9, 2.3], [-0.4, 1.2]])
    cart_poly = np.array([[0, 0], [1.6, -0.6], [2.6, 1.2], [1.0, 2.6], [-1.0, 1.4]])

    axes[0].set_title("Map region vs. cartogram region")
    _fill_poly(axes[0], map_poly, COLOR_MAP, label="Map $P_i$")
    _fill_poly(axes[0], cart_poly, COLOR_CARTOGRAM, label="Cartogram $P_i'$")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    # normalize both (unit area, centred at origin) and overlay
    n_map = _normalize_shape(map_poly)
    n_cart = _normalize_shape(cart_poly)
    sp_map = ShapelyPolygon(n_map)
    sp_cart = ShapelyPolygon(n_cart)
    sym_diff = sp_map.symmetric_difference(sp_cart)
    inter = sp_map.intersection(sp_cart)

    axes[1].set_title(r"Normalised overlay: $\delta_i = \frac{1}{2}\,\mathrm{area}(P_i \triangle P_i')$")

    def _plot_shapely(ax, geom, color, alpha):
        if geom.is_empty:
            return
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=color, alpha=alpha)

    _plot_shapely(axes[1], sym_diff, COLOR_BAD, 0.45)   # the non-overlap = the "error"
    _plot_shapely(axes[1], inter, COLOR_GOOD, 0.45)     # the shared part
    nx, ny = np.vstack([n_map, n_map[0]]).T
    axes[1].plot(nx, ny, color=COLOR_MAP, linewidth=1.4)
    nx2, ny2 = np.vstack([n_cart, n_cart[0]]).T
    axes[1].plot(nx2, ny2, color=COLOR_CARTOGRAM, linewidth=1.4)

    delta = 0.5 * sym_diff.area
    legend_handles = [
        mpatches.Patch(color=COLOR_GOOD, alpha=0.45, label="Overlap (agreement)"),
        mpatches.Patch(color=COLOR_BAD, alpha=0.45, label="Symmetric difference (error)"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)
    axes[1].text(0.5, -0.06, rf"$\delta_i \approx {delta:.2f}$", ha="center", va="top",
                 fontsize=12, color=COLOR_BAD, transform=axes[1].transAxes)

    # fig.suptitle("Shape deformation (Hamming distance): both shapes recentred & rescaled to unit area first",
    #               fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 3. Positional fidelity: angular + orthogonal error — Section 4.4.3
# ----------------------------------------------------------------------
def explain_positional_fidelity():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    ci_orig, cj_orig = np.array([0, 0]), np.array([3, 1])
    ci_new, cj_new = np.array([0, 0]), np.array([2.6, -0.6])  # j has drifted below i

    ax = axes[0]
    ax.set_title(r"Angular orientation error $\theta_{ij}$")
    ax.scatter(*ci_orig, color="black", zorder=5)
    ax.scatter(*cj_orig, color="black", zorder=5)
    ax.annotate("", xy=cj_orig, xytext=ci_orig,
                arrowprops=dict(arrowstyle="->", color=COLOR_MAP, lw=2))
    ax.scatter(*cj_new, color=COLOR_CARTOGRAM, zorder=5)
    ax.annotate("", xy=cj_new, xytext=ci_new,
                arrowprops=dict(arrowstyle="->", color=COLOR_CARTOGRAM, lw=2))
    ax.text(*ci_orig, "  $c_i$", fontsize=10)
    ax.text(*cj_orig, "  $c_j$ (map)", fontsize=10, color=COLOR_MAP)
    ax.text(*cj_new, "  $c_j'$ (cartogram)", fontsize=10, color=COLOR_CARTOGRAM)
    ang_orig = np.degrees(np.arctan2(*(cj_orig - ci_orig)[::-1]))
    ang_new = np.degrees(np.arctan2(*(cj_new - ci_new)[::-1]))
    theta = abs(ang_orig - ang_new)
    ax.text(1.2, 1.6, rf"$\theta_{{ij}} \approx {theta:.0f}^\circ$", color=COLOR_BAD, fontsize=12)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2.5)

    ax = axes[1]
    ax.set_title(r"Orthogonal (relative-direction) error $\rho$")
    ax.axhline(0, color="lightgray", lw=1)
    ax.axvline(0, color="lightgray", lw=1)
    ax.scatter(*ci_orig, color="black", zorder=5)
    ax.scatter(*cj_orig, color=COLOR_MAP, zorder=5)
    ax.scatter(*cj_new, color=COLOR_CARTOGRAM, zorder=5)
    ax.text(*ci_orig, "  $c_i$", fontsize=10)
    ax.text(*cj_orig, "  $c_j$: N & E of $c_i$ (map)", fontsize=9, color=COLOR_MAP)
    ax.text(*cj_new, "  $c_j'$: now S & E of $c_i$ (cartogram)", fontsize=9, color=COLOR_CARTOGRAM)
    ax.text(1.0, -1.7, r"sign of $\Delta y$ flipped $\Rightarrow$ counted in $\rho$", color=COLOR_BAD, fontsize=10)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-2, 2.5)

    # fig.suptitle("Positional fidelity: does region j stay in the same direction from region i?",
                  # fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 4. Adjacency (topological) error — Section 4.4.4
# ----------------------------------------------------------------------
def explain_adjacency_error():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3)

    nodes = {"A": (0, 1.5), "B": (1.5, 2.5), "C": (1.5, 0.5), "D": (3, 1.5)}
    edges_map = {("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")}
    edges_cart = {("A", "B"), ("B", "C"), ("B", "D")}  # A-C lost, B-D gained

    def draw_graph(ax, edges, ref_edges, title):
        ax.set_title(title)
        for (u, v) in edges:
            p1, p2 = nodes[u], nodes[v]
            in_ref = (u, v) in ref_edges or (v, u) in ref_edges
            color = COLOR_GOOD if in_ref else COLOR_BAD
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=2.5, zorder=1)
        for name, (x, y) in nodes.items():
            ax.scatter(x, y, s=350, color=COLOR_MAP, zorder=3)
            ax.text(x, y, name, ha="center", va="center", color="white", fontweight="bold", zorder=4)

    draw_graph(axes[0], edges_map, edges_map, "Map adjacency graph $E_{\\mathrm{orig}}$")
    draw_graph(axes[1], edges_cart, edges_map, "Cartogram adjacency graph $E_{\\mathrm{cart}}$")

    handles = [
        Line2D([0], [0], color=COLOR_GOOD, lw=2.5, label="Adjacency preserved"),
        Line2D([0], [0], color=COLOR_BAD, lw=2.5, label="Adjacency lost / gained"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    inter = len(edges_map & edges_cart)
    union = len(edges_map | edges_cart)
    tau = 1 - inter / union
    # fig.suptitle(rf"Adjacency error $\tau = 1 - |E_{{\mathrm{{cart}}}}\cap E_{{\mathrm{{orig}}}}| / |E_{{\mathrm{{cart}}}}\cup E_{{\mathrm{{orig}}}}| \approx {tau:.2f}$",
    #              fontsize=12, y=1.04)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 5. Overlap / self-intersection — Section 4.4.5
# ----------------------------------------------------------------------
def explain_overlap():
    fig, ax = _new_ax(figsize=(6, 5.5))#, title="Overlap: two regions occupying the same space")
    poly_a = np.array([[0, 0], [2.2, 0.2], [2.0, 2.2], [-0.2, 2.0]])
    poly_b = np.array([[1.2, 0.6], [3.4, 0.9], [3.0, 2.6], [1.0, 2.4]])
    sp_a, sp_b = ShapelyPolygon(poly_a), ShapelyPolygon(poly_b)
    overlap = sp_a.intersection(sp_b)

    _fill_poly(ax, poly_a, COLOR_MAP, alpha=0.4, label="$P_i$")
    _fill_poly(ax, poly_b, COLOR_CARTOGRAM, alpha=0.4, label="$P_j$")
    if not overlap.is_empty:
        xs, ys = overlap.exterior.xy
        ax.fill(xs, ys, color=COLOR_BAD, alpha=0.75, label="overlap area")

    total_area = _shoelace_area(poly_a) + _shoelace_area(poly_b)
    phi = overlap.area / total_area
    ax.text(0.5, -0.04, rf"$\Phi_{{\mathrm{{overlap}}}} = \dfrac{{\sum_{{i<j}} \mathrm{{area}}(P_i \cap P_j)}}{{\sum_i \mathrm{{area}}(P_i)}} \approx {phi:.2f}$",
            ha="center", va="top", fontsize=11, color=COLOR_BAD, transform=ax.transAxes)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 6. Gap error — Section 4.4.6
# ----------------------------------------------------------------------
def explain_gap_error():
    fig, ax = _new_ax(figsize=(6, 5.5))#, title="Gap: previously-touching regions pulling apart")

    def square(cx, cy, s=1.0):
        return np.array([[cx - s/2, cy - s/2], [cx + s/2, cy - s/2],
                          [cx + s/2, cy + s/2], [cx - s/2, cy + s/2]])

    regions = [square(0, 0), square(1.3, 0.05), square(0.05, 1.35), square(1.35, 1.4)]
    shapely_polys = [ShapelyPolygon(r) for r in regions]
    hull = unary_union(shapely_polys).convex_hull
    sum_area = sum(p.area for p in shapely_polys)
    gap_area = hull.area - sum_area

    hx, hy = hull.exterior.xy
    ax.plot(hx, hy, color=COLOR_MAP, linestyle="--", linewidth=1.4, label="outer boundary of union")

    # Shade the hull minus the regions to visualise the gap. hull.difference(...) is a
    # polygon *with holes* where the regions sit, so first fill its plain exterior, then
    # redraw the regions on top (rather than trying to render the holes directly) so the
    # regions are not painted over by the gap shading.
    gap_geom = hull.difference(unary_union(shapely_polys))
    if not gap_geom.is_empty:
        geoms = [gap_geom] if gap_geom.geom_type == "Polygon" else list(gap_geom.geoms)
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=COLOR_GAP, alpha=0.35, hatch="//", label="gap area", zorder=1)

    for r in regions:
        _fill_poly(ax, r, COLOR_CARTOGRAM, alpha=0.85)

    ax.text(0.5, -0.04, rf"Gap $= \mathrm{{area(outer\ boundary)}} - \sum_i \mathrm{{area}}(P_i) \approx {gap_area:.2f}$",
            ha="center", va="top", fontsize=11, color=COLOR_BAD, transform=ax.transAxes)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 7. Spatial deformation — Section 2.4, criterion 5
# (regions placed close to their original geographic position — an overall,
# per-region positional-drift criterion, distinct from the pairwise angular/
# orthogonal metrics of Section 4.4.3 which look at relative direction only)
# ----------------------------------------------------------------------
def explain_spatial_deformation():
    fig, ax = _new_ax(figsize=(6.5, 5.5),
                       title="Spatial deformation: has each region drifted from its original place?")

    rng = np.random.default_rng(7)
    orig = np.array([[0, 0], [2, 0.3], [1, 1.8], [-1.5, 1.2], [-0.8, -1.3], [1.6, -1.4]])
    # cartogram centres: mostly small drift, one region (index 2) drifts far
    drift = np.array([[0.15, -0.1], [-0.2, 0.15], [1.6, 1.1], [0.3, -0.2], [-0.15, 0.1], [0.2, 0.2]])
    new = orig + drift

    for o, n in zip(orig, new):
        ax.annotate("", xy=n, xytext=o,
                    arrowprops=dict(arrowstyle="->", color=COLOR_BAD, lw=1.6, alpha=0.85))
    ax.scatter(orig[:, 0], orig[:, 1], color=COLOR_MAP, s=90, zorder=5, label="Original position $c_i$")
    ax.scatter(new[:, 0], new[:, 1], color=COLOR_CARTOGRAM, s=90, zorder=5, label="Cartogram position $c_i'$")

    diag = np.linalg.norm(orig.max(axis=0) - orig.min(axis=0))
    mean_disp = np.mean(np.linalg.norm(drift, axis=1))
    ax.text(0.5, -0.04,
            rf"mean displacement $\dfrac{{1}}{{n}}\sum_i \|c_i' - c_i\|\ /\ \mathrm{{map\ diagonal}} \approx {mean_disp/diag:.2f}$",
            ha="center", va="top", fontsize=11, color=COLOR_BAD, transform=ax.transAxes)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 8. Global shape preservation — Section 2.4, criterion 6
# (does the overall silhouette / configuration of the whole map survive,
# as opposed to criterion 2 / Hamming distance which looks at one region
# at a time)
# ----------------------------------------------------------------------
def explain_global_shape_preservation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    # a jagged "country-like" outline standing in for the whole map's silhouette
    map_outline = np.array([
        [0.0, 0.0], [1.2, -0.3], [2.0, 0.2], [1.8, 1.1], [2.6, 1.6],
        [2.2, 2.4], [1.0, 2.1], [0.6, 2.8], [-0.4, 2.2], [-0.6, 1.0], [-0.2, 0.4]
    ])
    # a rounder, blobbier outline standing in for the cartogram's overall silhouette
    theta = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    cart_outline = np.column_stack([1 + 1.15 * np.cos(theta), 1.1 + 1.15 * np.sin(theta)])
    cart_outline += 0.12 * np.sin(3 * theta)[:, None]

    axes[0].set_title("Whole-map outline: map vs. cartogram")
    _fill_poly(axes[0], map_outline, COLOR_MAP, label="Map silhouette")
    _fill_poly(axes[0], cart_outline, COLOR_CARTOGRAM, label="Cartogram silhouette")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    n_map = _normalize_shape(map_outline)
    n_cart = _normalize_shape(cart_outline)
    sp_map, sp_cart = ShapelyPolygon(n_map), ShapelyPolygon(n_cart)
    sym_diff = sp_map.symmetric_difference(sp_cart)
    inter = sp_map.intersection(sp_cart)

    axes[1].set_title("Normalised overlay of overall configuration")

    def _plot_shapely(ax, geom, color, alpha):
        if geom.is_empty:
            return
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for g in geoms:
            xs, ys = g.exterior.xy
            ax.fill(xs, ys, color=color, alpha=alpha)

    _plot_shapely(axes[1], sym_diff, COLOR_BAD, 0.45)
    _plot_shapely(axes[1], inter, COLOR_GOOD, 0.45)
    mx, my = np.vstack([n_map, n_map[0]]).T
    axes[1].plot(mx, my, color=COLOR_MAP, linewidth=1.4)
    cx_, cy_ = np.vstack([n_cart, n_cart[0]]).T
    axes[1].plot(cx_, cy_, color=COLOR_CARTOGRAM, linewidth=1.4)

    global_delta = 0.5 * sym_diff.area
    axes[1].text(0.5, -0.06, rf"global shape mismatch $\approx {global_delta:.2f}$",
                 ha="center", va="top", fontsize=11, color=COLOR_BAD, transform=axes[1].transAxes)

    # fig.suptitle("Global shape preservation: does the overall map configuration stay recognisable?",
    #              fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 2. Shape deformation — Section 2.4, criterion 2
# ("Regions should resemble their geographic shape" — the general outline-
# resemblance criterion; formally this is exactly the Hamming distance of
# Section 4.4.2 / explain_hamming_distance(), this figure is the more
# intuitive companion showing *why* a rounded-off outline reads as worse
# shape deformation even before computing that number)
# ----------------------------------------------------------------------
def explain_shape_deformation():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    # a region with two distinctive local features: a sharp inward notch and a
    # pointed peninsula, both easy for a human reader to recognise
    base = np.array([
        [0, 0], [1.0, -0.1], [1.3, 0.6], [1.9, 0.5],  # peninsula tip at (1.9, 0.5)
        [1.5, 0.9], [1.6, 1.6], [0.9, 1.3], [0.8, 0.9],  # notch corner at (0.8, 0.9)
        [0.3, 1.1], [-0.3, 0.6]
    ])
    peninsula_idx, notch_idx = 3, 7

    def deform(points, feature_strength):
        """feature_strength=1 keeps corners sharp, 0 rounds everything toward
        the shape's own centroid (mimicking what a Dorling/mosaic-style
        simplification does to local detail)."""
        c = points.mean(axis=0)
        smoothed = (points + np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) / 3
        return feature_strength * points + (1 - feature_strength) * (0.6 * smoothed + 0.4 * c)

    preserved = deform(base, 0.85)
    lost = deform(base, 0.05)

    axes[0].set_title("Low shape deformation")
    _fill_poly(axes[0], preserved, COLOR_GOOD, alpha=0.35)
    axes[0].scatter(*preserved[peninsula_idx], color=COLOR_GOOD, s=60, zorder=5)
    axes[0].scatter(*preserved[notch_idx], color=COLOR_GOOD, s=60, zorder=5)
    axes[0].text(preserved[peninsula_idx, 0] + 0.1, preserved[peninsula_idx, 1], "peninsula\nkept", fontsize=8, color=COLOR_GOOD)
    axes[0].text(preserved[notch_idx, 0] - 1.0, preserved[notch_idx, 1], "notch\nkept", fontsize=8, color=COLOR_GOOD)

    axes[1].set_title("High shape deformation")
    _fill_poly(axes[1], lost, COLOR_BAD, alpha=0.35)
    axes[1].text(0.5, -0.04, "outline rounded into a generic blob\n(no longer resembles the geographic shape)",
                 ha="center", va="top", fontsize=9, color=COLOR_BAD, transform=axes[1].transAxes)

    # fig.suptitle("Shape deformation: does the region still resemble its geographic outline?",
    #              fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 7. Local shape — Section 2.4, criterion 7
# ("Regions on identifiable positions should match" — distinct from shape
# deformation above: this asks whether specific, identifiable landmark
# points *inside* a region end up in the corresponding relative place after
# the transformation, not just whether the aggregate outline/area matches)
# ----------------------------------------------------------------------
def explain_local_shape():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_xlim(-0.3, 2.3)
        ax.set_ylim(-0.3, 2.3)

    # an irregular input region with three identifiable landmark points at
    # fixed *relative* (fx, fy) positions inside its bounding box
    input_poly = np.array([
        [0.2, 0.3], [1.4, 0.0], [2.0, 0.9], [1.7, 1.9], [0.6, 2.0], [-0.1, 1.1]
    ])
    landmarks = {"orange": (0.62, 0.78), "steelblue": (0.30, 0.42), "seagreen": (0.75, 0.28)}
    marker_color = {"orange": "#F26419", "steelblue": "#2E86AB", "seagreen": "#4C9A2A"}

    def bbox(points):
        return points.min(axis=0), points.max(axis=0)

    lo, hi = bbox(input_poly)

    def place(fx, fy, lo, hi):
        return np.array([lo[0] + fx * (hi[0] - lo[0]), lo[1] + fy * (hi[1] - lo[1])])

    axes[0].set_title("Input region")
    _fill_poly(axes[0], input_poly, COLOR_MAP, alpha=0.35)
    for name, (fx, fy) in landmarks.items():
        p = place(fx, fy, lo, hi)
        axes[0].scatter(*p, color=marker_color[name], s=90, zorder=5, edgecolor="black", linewidth=0.6)

    # target shape used for both the "good" and "bad" cartogram cells (e.g. a
    # Demers-style square region, split visually into a grid so relative
    # position within it is easy to read off)
    target = np.array([[0.1, 0.1], [2.0, 0.1], [2.0, 2.0], [0.1, 2.0]])
    t_lo, t_hi = bbox(target)

    def draw_grid(ax, ncols, nrows, colors):
        w = (t_hi[0] - t_lo[0]) / ncols
        h = (t_hi[1] - t_lo[1]) / nrows
        k = 0
        for r in range(nrows):
            for c in range(ncols):
                cell = np.array([
                    [t_lo[0] + c * w, t_lo[1] + r * h],
                    [t_lo[0] + (c + 1) * w, t_lo[1] + r * h],
                    [t_lo[0] + (c + 1) * w, t_lo[1] + (r + 1) * h],
                    [t_lo[0] + c * w, t_lo[1] + (r + 1) * h],
                ])
                _fill_poly(ax, cell, colors[k % len(colors)], alpha=0.55, lw=0.8)
                k += 1

    # "Good": landmarks keep the same relative (fx, fy) position in the new
    # shape, so their spatial arrangement relative to each other is preserved
    axes[1].set_title("Good: identifiable positions match")
    draw_grid(axes[1], 2, 1, ["#B9BEC7", COLOR_CARTOGRAM])
    for name, (fx, fy) in landmarks.items():
        p = place(fx, fy, t_lo, t_hi)
        axes[1].scatter(*p, color=marker_color[name], s=90, zorder=5, edgecolor="black", linewidth=0.6)

    # "Bad": landmarks are scrambled to unrelated relative positions -- the
    # region's overall outline/area could still match, but the identifiable
    # points no longer correspond to where they should be
    axes[2].set_title("Bad: identifiable positions scrambled")
    draw_grid(axes[2], 4, 3, ["#B9BEC7", COLOR_CARTOGRAM])
    scrambled = {"orange": (0.15, 0.85), "steelblue": (0.85, 0.15), "seagreen": (0.45, 0.55)}
    for name, (fx, fy) in scrambled.items():
        p = place(fx, fy, t_lo, t_hi)
        axes[2].scatter(*p, color=marker_color[name], s=90, zorder=5, edgecolor="black", linewidth=0.6)

    # fig.suptitle("Local shape: do identifiable landmark points end up in the corresponding place?",
    #              fontsize=12, y=1.04)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 10. Complexity — Section 2.4, criterion 8
# (the framework's own complexity proxy is mean/max vertex count per
# region, Section 4.4.7 — more vertices, more visual clutter, harder to
# read at a glance)
# ----------------------------------------------------------------------
def explain_complexity():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    rng = np.random.default_rng(3)
    n_high = 48
    theta_h = np.linspace(0, 2 * np.pi, n_high, endpoint=False)
    r_h = 1 + 0.18 * np.sin(theta_h * 7) + 0.08 * rng.normal(size=n_high)
    high_complexity = np.column_stack([r_h * np.cos(theta_h), r_h * np.sin(theta_h)])

    n_low = 6
    theta_l = np.linspace(0, 2 * np.pi, n_low, endpoint=False)
    low_complexity = np.column_stack([np.cos(theta_l), np.sin(theta_l)])

    axes[0].set_title(f"High complexity ({n_high} vertices)")
    _fill_poly(axes[0], high_complexity, COLOR_BAD, alpha=0.4)
    axes[0].text(0.5, -0.04, "post-processed / raw coastline-like outline\nharder to read at a glance",
                 ha="center", va="top", fontsize=9, color=COLOR_BAD, transform=axes[0].transAxes)

    axes[1].set_title(f"Low complexity ({n_low} vertices)")
    _fill_poly(axes[1], low_complexity, COLOR_GOOD, alpha=0.4)
    axes[1].text(0.5, -0.04, "simplified shape (e.g. Dorling/Demers style)\neasier to read at a glance",
                 ha="center", va="top", fontsize=9, color=COLOR_GOOD, transform=axes[1].transAxes)

    # fig.suptitle("Complexity: vertex count as a proxy for how easy the cartogram is to read",
    #              fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 11. Data ink ratio — Section 2.4, criterion 9
# (Tufte-style proportion of the canvas that carries information vs. empty
# void — no formal equation in this thesis, included here for conceptual
# completeness against the Section 2.4 criteria list)
# ----------------------------------------------------------------------
def explain_data_ink_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

    frame = np.array([[0, 0], [4, 0], [4, 3], [0, 3]])

    def small_circle(cx, cy, r):
        t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])

    # (a) low ink ratio: small scattered shapes, mostly empty canvas
    scattered = [small_circle(0.8, 2.2, 0.35), small_circle(1.9, 2.5, 0.25),
                 small_circle(1.3, 0.9, 0.3), small_circle(2.7, 1.4, 0.2)]

    axes[0].set_title("Low data-ink ratio")
    axes[0].plot(*np.vstack([frame, frame[0]]).T, color="black", lw=1)
    total_ink = 0.0
    for s in scattered:
        _fill_poly(axes[0], s, COLOR_CARTOGRAM, alpha=0.55)
        total_ink += _shoelace_area(s)
    frame_area = _shoelace_area(frame)
    axes[0].text(0.5, -0.04, rf"ink / canvas $\approx {total_ink/frame_area:.2f}$  (mostly void)",
                 ha="center", va="top", fontsize=10, color=COLOR_BAD, transform=axes[0].transAxes)

    # (b) high ink ratio: tiled regions filling the canvas edge-to-edge
    axes[1].set_title("High data-ink ratio")
    axes[1].plot(*np.vstack([frame, frame[0]]).T, color="black", lw=1)
    tile_ink = 0.0
    rng2 = np.random.default_rng(1)
    for i in range(4):
        for j in range(3):
            jitter = 0.03 * rng2.normal(size=2)
            tile = np.array([[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]], dtype=float) + jitter
            _fill_poly(axes[1], tile, COLOR_CARTOGRAM, alpha=0.55)
            tile_ink += _shoelace_area(tile)
    axes[1].text(0.5, -0.04, rf"ink / canvas $\approx {tile_ink/frame_area:.2f}$  (little void)",
                 ha="center", va="top", fontsize=10, color=COLOR_GOOD, transform=axes[1].transAxes)

    # fig.suptitle("Data-ink ratio: how much of the canvas is carrying information vs. empty space",
    #              fontsize=12, y=1.03)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
FIGURES = {
    "Cartographic": explain_cartographic_error,
    "ShapeDeformation": explain_shape_deformation,
    "HammingDistance": explain_hamming_distance,
    "PositionalFidelity": explain_positional_fidelity,
    "AdjacencyError": explain_adjacency_error,
    "Overlap": explain_overlap,
    "GapError": explain_gap_error,
    "SpatialDeformation": explain_spatial_deformation,
    "GlobalShapePreservation": explain_global_shape_preservation,
    "LocalShape": explain_local_shape,
    "Complexity": explain_complexity,
    "DataInkRatio": explain_data_ink_ratio,
}


def save_all(outdir="qc_figures", dpi=200, formats=("png", "svg", "pdf")):
    """Render every explainer figure and save it under outdir, ready for
    \\includegraphics in the thesis (Section 2.4 / 4.4 are natural homes)."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fig = fn()
        for fmt in formats:
            path = os.path.join(outdir, f"{name}.{fmt}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
    return paths


if __name__ == "__main__":
    saved = save_all()
    print(f"Saved {len(saved)} files:")
    for p in saved:
        print(" -", p)
