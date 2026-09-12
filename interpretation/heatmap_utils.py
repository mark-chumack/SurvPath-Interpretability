"""Shared rendering for the pathway -> patch attention heatmaps.

Both ``interpret_survpath.py`` and ``render_gene_pathway_map.py`` paint the same
thing: one scalar per WSI patch, blended onto a slide thumbnail. This module
holds that painting so the two stay consistent, and fixes the three things that
made the first-generation overlays unreadable:

1.  TILE FOOTPRINT.  The old code sized each painted tile as
    ``int(patch_size / ds)`` from a *hard-coded* ``patch_size`` (256).  When the
    features were extracted on a coarser grid (CLAM ``patch_level > 0``, so a
    1024 px level-0 stride) every tile covered 1/16 of its real area, leaving a
    regular lattice of tiny dots separated by unpainted slide.  We now infer the
    stride from the coords grid itself (`infer_patch_stride`) so tiles tile the
    tissue seamlessly.

2.  DYNAMIC RANGE.  ``cross_attn_pathways`` is *pre-softmax* (raw q.k dot
    products, see models/layers/cross_attention.py:90).  Its distribution is
    roughly bell-shaped, so min-max scaling parks almost every patch in the
    middle of the colormap -> uniform mid-tone confetti with no visible
    structure.  `normalize_scores` defaults to a rank (percentile) transform,
    which spreads the tiles uniformly over the ramp so spatial structure
    actually shows, and reports the raw range in the caption so nothing is
    hidden by the rescaling.

3.  COLOR + LEGEND.  ``jet`` is a rainbow ramp: non-monotone lightness, so
    magnitude is unreadable and it never had a legend at all.  Magnitude gets a
    single-hue sequential ramp (light -> dark), signed maps get a two-hue
    diverging ramp with a neutral gray midpoint, every figure carries a
    colorbar, and `color_caption` writes out in words what each end of the ramp
    means.

...and then three more that only showed up once real slides were rendered:

4.  THE OVERLAY LIED ABOUT ITSELF.  With ``norm=rank`` every tile gets a
    distinct percentile by construction, so the alpha ramp tints essentially all
    the tissue — while the caption claimed "unpainted H&E means the model barely
    looked here". Two fixes: `paint_attention` now returns the fraction of
    tissue it actually inked, and `color_caption` describes *that measurement*
    instead of an assumption. `render="mosaic"` (the paper's A_P->H look) drops
    the H&E blend entirely and draws opaque tiles on a flat canvas, so there is
    no "unpainted tissue" claim left to get wrong.

5.  IS THERE SIGNAL AT ALL?  Raw pre-softmax attention with a median near 0 and
    a narrow spread can still be rank-stretched into a vivid picture of nothing.
    `spatial_diagnostics` measures whether a map is spatially structured
    (Moran's I on the patch grid against a permutation null) and says so in the
    caption; below ~2 sigma the figure is labelled as noise rather than shipped
    looking confident.

6.  EVERY PATHWAY LOOKED THE SAME.  Most of A_P->H is a per-patch component that
    all P pathways share — rows typically correlate above rho=0.9 — so the
    panels are near-duplicates.  That is a true property of the raw attention
    rather than a plotting bug, so the renderers keep plotting rows verbatim and
    `specificity_report` measures the redundancy for the caption to state
    outright, instead of letting the layout imply a localisation the model does
    not have.

Palette values are the documented sequential-blue and diverging blue<->red
instances from the project's data-viz reference; only the dark end of the red
arm is derived (scaled toward black in linear light) because the reference
documents red as a categorical anchor without a full ramp.
``render_cross_modal.py`` overrides all of that with ``jet`` deliberately, to
reproduce the published figure — see that module's header for the trade.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"          # figure background (light chart surface)
MOSAIC_BG = "#ffffff"        # canvas behind a tissue-only mosaic panel
INK = "#0b0b0b"              # primary ink: titles
INK_SECONDARY = "#52514e"    # secondary ink: captions
INK_MUTED = "#898781"        # muted ink: ticks, axis labels
HAIRLINE = "#e1e0d9"

# Diverging poles, reused for the attribution bars: red = pushes risk up,
# blue = pushes risk down. Same two hues as the diverging ramp, so a red tile
# and a red bar mean the same direction everywhere in the figure.
POLE_POS = "#c03a39"         # increases risk
POLE_NEG = "#256abf"         # decreases risk
NEUTRAL = "#f0efec"          # diverging midpoint / zero

# Third pole for the categorical contribution map: a tile the model leaned on
# heavily whose feature-level attributions cancel, so it has no net direction.
# Deliberately off the red<->blue axis — it is not "somewhere between raising
# and lowering risk", it is a different statement — and it is the reference
# green, so it does not read as a third point on the diverging ramp.
POLE_MIXED = "#1baf7a"       # high influence, no net direction

# Sequential: one hue, light -> dark (steps 100..700 of the reference blue ramp).
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b")

# Diverging: blue <-> red poles (read as opposite) with a neutral gray midpoint.
# Equal step count per arm; lightness is monotone out of the midpoint on both.
DIV_BLUE_RED = ("#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb",
                "#f0efec",
                "#f7cfce", "#f0a3a1", "#e34948", "#c03a39", "#9c2f2e", "#7d2827")

_NAMED_RAMPS = {
    # name -> (hex steps, is_diverging)
    "attention": (SEQ_BLUE, False),
    "attention-signed": (DIV_BLUE_RED, True),
}

#: Categorical hues, for figures where color names *which* pathway rather than
#: *how much*. Ordered for maximum separation between the first few, which are
#: the ones a reader has to tell apart. Six is the practical ceiling for
#: identifying a category by hue alone, so callers cap their category count at
#: ``len(CATEGORICAL)`` rather than cycling and drawing two pathways the same.
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")

#: cmap names that carry meaning correctly here. Anything else is passed through
#: to matplotlib so old commands keep working, with a warning.
DEFAULT_CMAP = "attention"
DEFAULT_SIGNED_CMAP = "attention-signed"


RAINBOW_CMAPS = ("jet", "rainbow", "gist_rainbow", "hsv", "nipy_spectral", "turbo")


def resolve_cmap(name=DEFAULT_CMAP, warn=True):
    """Return a matplotlib Colormap for ``name``.

    Our own single-hue / diverging ramps are built here; any other name falls
    through to matplotlib's registry (so ``--cmap magma`` still works), but
    rainbow ramps get a warning because non-monotone lightness makes magnitude
    unreadable.

    ``warn=False`` silences that for a caller who has chosen a rainbow ramp
    deliberately — render_cross_modal does, to reproduce the published figure —
    and is expected to say so in its own caption instead.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib

    if name in _NAMED_RAMPS:
        steps, _ = _NAMED_RAMPS[name]
        return LinearSegmentedColormap.from_list(name, list(steps), N=256)

    if warn and name in RAINBOW_CMAPS:
        print(f"[warn] cmap '{name}' is a rainbow ramp: its lightness is not "
              f"monotone, so readers cannot rank two tiles by color. Prefer "
              f"'{DEFAULT_CMAP}' (sequential) or '{DEFAULT_SIGNED_CMAP}' "
              f"(diverging, for signed scores).")
    return matplotlib.colormaps[name]


def is_diverging(name):
    return _NAMED_RAMPS.get(name, (None, False))[1]


def _hex_at(name, frac):
    """Hex color at position ``frac`` (0..1) of a ramp — for caption text."""
    from matplotlib.colors import to_hex
    # warn=False: whoever picked the ramp was already warned at paint time.
    return to_hex(resolve_cmap(name, warn=False)(float(np.clip(frac, 0.0, 1.0))))


# --------------------------------------------------------------------------- #
# what the map is *of*
#
# The painting is quantity-agnostic — one scalar per patch — but the legend and
# caption are not: "how strongly this pathway attends here" is a false statement
# about an integrated-gradients map, which measures contribution to the risk
# score and has a meaningful zero and sign. So the wording travels with the
# scores as a MapSemantics rather than being baked into the renderer.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MapSemantics:
    """Wording for one kind of per-patch score. Fields are format templates.

    Available placeholders: ``{subject}`` (e.g. "pathway 42"), ``{tile}``
    (" per 1024 px tile (~32 px on this thumbnail)", possibly empty),
    ``{lo}`` / ``{mid}`` / ``{hi}`` (hex colors sampled from the active ramp).
    """
    quantity: str        # noun used in colorbar labels and the raw-range line
    lead: str            # "COLOR = {lead}, <how it was normalised>."
    diverging: str       # what the two poles of a diverging ramp mean
    sequential: str      # what light -> dark means on a sequential ramp
    low_phrase: str      # "...so paler H&E is {low_phrase}."
    hot_word: str        # "Read only the darkest tiles as '{hot_word}'"


#: Pathway -> patch cross-attention (the default; wording unchanged).
ATTENTION_SEMANTICS = MapSemantics(
    quantity="cross-attention",
    lead="how strongly {subject} attends to each{tile} of tissue",
    diverging="Blue {lo} = the pathway is anti-aligned with that tile; neutral "
              "gray {mid} = no relationship (score 0); red {hi} = strongly "
              "aligned. Saturation of either hue encodes magnitude.",
    sequential="Pale blue {lo} = the weakest-attended tissue; deepening blue "
               "through {mid} = progressively more attention; dark navy {hi} = "
               "the tissue this pathway attends to most.",
    low_phrase="tissue this pathway attends to least",
    hot_word="attended",
)

#: Integrated gradients of the risk score w.r.t. the patch features. Signed and
#: zero-referenced: the sign is the direction the tile moves predicted risk, so
#: the diverging wording is the honest default here.
IG_SEMANTICS = MapSemantics(
    quantity="IG attribution to risk",
    lead="how much each{tile} of tissue contributed to this case's predicted "
         "risk, by integrated gradients of the risk score w.r.t. that tile's "
         "features ({subject})",
    diverging="Blue {lo} = this tile pushes predicted risk DOWN (protective for "
              "this case); neutral gray {mid} = no net contribution "
              "(attribution 0); red {hi} = pushes predicted risk UP. Saturation "
              "of either hue encodes how much. Attributions sum, over all tiles "
              "and all pathway inputs, to the risk gap between this case and the "
              "baseline.",
    sequential="Pale blue {lo} = the tissue that moved the risk prediction "
               "least; deepening blue through {mid} = progressively more "
               "influence; dark navy {hi} = the tissue that moved it most. This "
               "ramp shows magnitude only — it does not say which direction.",
    low_phrase="tissue that barely moved this case's risk prediction",
    hot_word="contributing",
)


# --------------------------------------------------------------------------- #
# locating a case's coords (.h5) and slide (.svs)
#
# Both entry points resolve these the same way, so the lookup lives here:
# a case id (TCGA-AC-A23E) is NOT a file stem — the coords and slide files are
# named by *slide* id (TCGA-AC-A23E-01Z-00-DX1[.UUID][.svs]). The mapping from
# one to the other is the 'case_id' -> 'slide_id' column pair in the study
# metadata CSV, which is exactly what SurvivalDatasetFactory._get_patient_dict
# reads (datasets/dataset_survival.py:325) and what interpret_survpath.py then
# hands to find_case_file.
# --------------------------------------------------------------------------- #
SLIDE_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")


def find_case_file(directory, slide_id, exts):
    """'<slide_id stripped of .svs><ext>' if it exists, else None."""
    if not directory:
        return None
    stem = str(slide_id)
    if stem.endswith(".svs"):
        stem = stem[: -len(".svs")]
    for ext in exts:
        cand = os.path.join(directory, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def slide_ids_for_case(label_file, case_id):
    """The exact slide ids the dataset would hand back for one case.

    This mirrors SurvivalDatasetFactory step for step, so the stems here are
    byte-identical to the ones interpret_survpath.py looks up:

      _setup_metadata_and_labels  read_csv(label_file, low_memory=False)
      _clean_label_data           BRCA: keep only the IDC subtype
      _get_patient_dict           set_index('case_id').loc[case, 'slide_id'],
                                  a bare string becoming a 1-element array

    Returns [] when the file, the columns, or the case are absent, so the caller
    can fall back and say so.
    """
    if not label_file:
        return []
    if not os.path.isfile(label_file):
        print(f"[warn] label file not found: {label_file}")
        return []
    import pandas as pd

    label_data = pd.read_csv(label_file, low_memory=False)
    for col in ("case_id", "slide_id"):
        if col not in label_data.columns:
            print(f"[warn] {label_file} has no '{col}' column.")
            return []

    # dataset_survival.py:234 — same guard, so we keep the same rows it keeps.
    if "oncotree_code" in label_data.columns and "IDC" in label_data["oncotree_code"]:
        label_data = label_data[label_data["oncotree_code"] == "IDC"]

    try:
        slide_ids = label_data.set_index("case_id").loc[case_id, "slide_id"]
    except KeyError:
        return []
    if isinstance(slide_ids, str):                    # single-slide case
        return [slide_ids]
    return [str(s) for s in np.asarray(slide_ids).ravel().tolist()]


def check_case_files(case_id, slide_ids, coords_dir, slide_dir):
    """Verify the coords (.h5) and slide files exist for this case's slide_ids.

    Returns True iff every slide_id resolves to BOTH an .h5 and a slide file.
    Logs one line per slide so a missing/mis-named file is obvious up front,
    rather than being silently skipped at paint time."""
    if not coords_dir and not slide_dir:
        print(f"[check] INTERP_COORDS_DIR / INTERP_SLIDE_DIR not set -> "
              f"skipping spatial-file check for case {case_id}.")
        return False

    all_ok = True
    print(f"[check] case {case_id}: {len(slide_ids)} slide(s)")
    for sid in slide_ids:
        h5 = find_case_file(coords_dir, sid, (".h5",))
        slide = find_case_file(slide_dir, sid, SLIDE_EXTS)
        ok = (h5 is not None) and (slide is not None)
        all_ok &= ok
        stem = str(sid)[:-4] if str(sid).endswith(".svs") else str(sid)
        print(f"   [{'ok ' if ok else 'MISS'}] {sid}")
        if h5 is None:
            print(f"          h5   : not found -> expected {coords_dir}/{stem}.h5")
        else:
            print(f"          h5   : {h5}")
        if slide is None:
            print(f"          slide: not found -> expected {slide_dir}/{stem}"
                  f"{{{','.join(SLIDE_EXTS)}}}")
        else:
            print(f"          slide: {slide}")
    print(f"[check] case {case_id}: "
          f"{'all files present' if all_ok else 'MISSING files (see above)'}")
    return all_ok


# --------------------------------------------------------------------------- #
# patch geometry
# --------------------------------------------------------------------------- #
def _modal_spacing(values):
    """Most common positive gap between consecutive unique coordinates."""
    u = np.unique(np.asarray(values, dtype=np.int64))
    if u.size < 2:
        return None
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return None
    vals, counts = np.unique(d, return_counts=True)
    return int(vals[np.argmax(counts)])


def infer_patch_stride(coords, fallback=256, declared=None, verbose=True):
    """Infer the level-0 patch stride (px) from the CLAM coords grid.

    CLAM writes one coord per patch on a regular grid, so the modal gap between
    unique x (and y) coordinates *is* the stride. Inferring it means the painted
    tiles cover the tissue seamlessly no matter what patch_size / patch_level
    produced the features — which is what a caller-supplied ``patch_size`` got
    wrong whenever features came from a downsampled level.

    ``declared`` (the caller's --patch-size, if any) is only used when the grid
    is too degenerate to infer from, but a large disagreement is reported since
    it usually means the wrong .h5 is paired with the features.
    """
    coords = np.asarray(coords)
    sx = _modal_spacing(coords[:, 0])
    sy = _modal_spacing(coords[:, 1])
    candidates = [s for s in (sx, sy) if s and s > 0]

    if not candidates:
        stride = int(declared or fallback)
        if verbose:
            print(f"[warn] could not infer patch stride from coords "
                  f"({coords.shape[0]} patches); using {stride} px.")
        return stride

    stride = int(min(candidates))
    if verbose:
        print(f"[info] inferred patch stride {stride} px at level 0 "
              f"(modal coord spacing: x={sx}, y={sy})")
        if declared and abs(declared - stride) > 1:
            print(f"[info] ...overriding the declared patch size ({declared} px). "
                  f"Painting {stride} px tiles so they tile the slide without "
                  f"gaps. Pass --patch-size/-INTERP_PATCH_SIZE only if the "
                  f"coords grid is irregular.")
    return stride


# --------------------------------------------------------------------------- #
# how much do these pathways' maps actually differ?
#
# The biggest term in q_p . k_n is how "attendable" patch n is at all, not which
# pathway p is asking, so rows of cross_attn_pathways come out highly
# correlated. Panels for five different pathways then look like five copies of
# one picture — which is an accurate rendering of the raw attention, and exactly
# why a figure that shows them owes the reader the correlation as a number.
# --------------------------------------------------------------------------- #
def _rank01(x):
    """Ranks of ``x`` rescaled to 0..1 (ties broken by order; fine here)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.empty(x.size, dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r / max(x.size - 1, 1)


@dataclass
class SpecificityReport:
    """Redundancy among the pathway maps a figure is about to show."""
    n_pathways: int
    mean_rho: float                  # mean pairwise Spearman between the maps
    min_rho: float
    max_rho: float
    shared_var_frac: float           # var(mean map) / mean var(single map)

    @property
    def verdict(self):
        if not np.isfinite(self.mean_rho):
            return "not measurable"
        if self.mean_rho >= 0.9:
            return "near-duplicates"
        if self.mean_rho >= 0.6:
            return "largely shared"
        if self.mean_rho >= 0.3:
            return "partly shared"
        return "largely distinct"

    def summary(self):
        if not np.isfinite(self.mean_rho):
            return ""
        s = (f"SPECIFICITY: across the {self.n_pathways} pathways shown, these "
             f"maps rank-correlate at rho={self.mean_rho:.2f} on average "
             f"(range {self.min_rho:.2f}-{self.max_rho:.2f}), and the map they "
             f"share accounts for {self.shared_var_frac:.0%} of a single "
             f"pathway's variance — {self.verdict}.")
        if self.mean_rho >= 0.6:
            s += (" Panels that look alike therefore ARE alike: the model is "
                  "pooling from largely the same patches for every pathway, so "
                  "do not read these as pathway-specific localisation.")
        return s


def specificity_report(cross_attn, idxs):
    """Measure the "all five pathways look identical" effect rather than hide it.

    Rank (Spearman) correlation, because the maps are rank-normalised before
    painting — rank agreement is precisely "will these two panels look alike"
    once the ramp is applied.
    """
    a = np.asarray(cross_attn, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected [P, N] cross-attention, got {a.shape}")
    idxs = [int(i) for i in idxs]
    rows = a[idxs]
    ranks = np.stack([_rank01(r) for r in rows]) if len(idxs) else np.empty((0, 0))

    rhos = [float(c) for i in range(len(idxs)) for j in range(i + 1, len(idxs))
            if np.isfinite(c := np.corrcoef(ranks[i], ranks[j])[0, 1])]
    nan = float("nan")

    shared = rows.mean(axis=0) if len(idxs) else np.zeros(0)
    row_var = float(np.mean([np.var(r) for r in rows])) if len(idxs) else 0.0
    shared_frac = float(np.clip(np.var(shared) / row_var, 0.0, 1.0)) if row_var else nan

    return SpecificityReport(
        n_pathways=len(idxs),
        mean_rho=float(np.mean(rhos)) if rhos else nan,
        min_rho=float(np.min(rhos)) if rhos else nan,
        max_rho=float(np.max(rhos)) if rhos else nan,
        shared_var_frac=shared_frac)


# --------------------------------------------------------------------------- #
# is this map signal, or a rescaled cloud of noise?
#
# Rank normalisation is monotone, so it turns pure noise into a full-range,
# vivid picture just as readily as it does a real signature — which means a
# figure that uses it owes the reader a check that rescaling cannot fool.
# Moran's I is that check: it asks whether neighbouring patches hold similar
# values, which noise does not do and a spatial signature does, and it is
# invariant to every monotone transform we apply downstream.
# --------------------------------------------------------------------------- #
@dataclass
class SpatialDiagnostics:
    n_patches: int
    n_pairs: int
    morans_i: float                  # -1..1; ~0 = spatially unstructured
    z: float                         # sigma above a shuffled-patch null
    raw_median: float
    raw_iqr: float

    @property
    def structured(self):
        return bool(np.isfinite(self.z) and self.z >= 2.0)

    @property
    def strength(self):
        if not np.isfinite(self.morans_i):
            return "not measurable"
        if not self.structured:
            return "no spatial structure"
        if self.morans_i < 0.15:
            return "weak spatial structure"
        if self.morans_i < 0.40:
            return "moderate spatial structure"
        return "strong spatial structure"

    def summary(self):
        if not np.isfinite(self.morans_i):
            return ("SIGNAL: spatial structure not measurable (too few adjacent "
                    "patches).")
        s = (f"SIGNAL: Moran's I = {self.morans_i:+.3f} ({self.z:+.1f} sigma vs. "
             f"a shuffled-patch null, {self.n_pairs} adjacent pairs); raw scores "
             f"span an IQR of {self.raw_iqr:.2f} about a median of "
             f"{self.raw_median:+.2f} — {self.strength}.")
        if not self.structured:
            s += (" Neighbouring patches are no more alike than randomly paired "
                  "ones, so what is drawn is the ramp stretching noise across "
                  "the slide, not a localised signature. Read it as no finding.")
        return s


def _grid_neighbour_pairs(coords, stride_l0):
    """Index pairs of 4-connected neighbours on the CLAM patch grid.

    Snapping coords to grid cells and looking up the right/down neighbour keeps
    this O(N) and exact for the regular grid CLAM writes — and lists each
    undirected pair once.
    """
    xy = np.asarray(coords, dtype=np.float64)
    s = float(stride_l0) or 1.0
    gx = np.rint((xy[:, 0] - xy[:, 0].min()) / s).astype(np.int64)
    gy = np.rint((xy[:, 1] - xy[:, 1].min()) / s).astype(np.int64)

    lookup = {cell: i for i, cell in enumerate(zip(gx.tolist(), gy.tolist()))}
    ii, jj = [], []
    for i, (a, b) in enumerate(zip(gx.tolist(), gy.tolist())):
        for cell in ((a + 1, b), (a, b + 1)):
            j = lookup.get(cell)
            if j is not None:
                ii.append(i)
                jj.append(j)
    return np.asarray(ii, dtype=np.int64), np.asarray(jj, dtype=np.int64)


def _morans_i(values, ii, jj):
    """Moran's I under a symmetric binary adjacency given as an edge list."""
    z = np.asarray(values, dtype=np.float64)
    z = z - z.mean()
    denom = float((z * z).sum())
    if denom <= 0 or ii.size == 0:
        return float("nan")
    # W = 2 * n_edges, and each undirected edge enters the double sum twice
    num = 2.0 * float((z[ii] * z[jj]).sum())
    return (z.size / (2.0 * ii.size)) * (num / denom)


def spatial_diagnostics(coords, values, stride_l0, n_perm=199, seed=0):
    """Is ``values`` spatially structured over the patch grid, or just noise?"""
    v = np.asarray(values, dtype=np.float64).ravel()
    ii, jj = _grid_neighbour_pairs(coords, stride_l0)
    q75, q25 = np.percentile(v, [75, 25])
    common = dict(n_patches=int(v.size), n_pairs=int(ii.size),
                  raw_median=float(np.median(v)), raw_iqr=float(q75 - q25))

    obs = _morans_i(v, ii, jj)
    if not np.isfinite(obs) or ii.size < 8:
        return SpatialDiagnostics(morans_i=obs, z=float("nan"), **common)

    rng = np.random.default_rng(seed)
    null = np.array([_morans_i(rng.permutation(v), ii, jj) for _ in range(n_perm)])
    sd = float(null.std())
    z = (obs - float(null.mean())) / sd if sd > 0 else float("nan")
    return SpatialDiagnostics(morans_i=float(obs), z=float(z), **common)


# --------------------------------------------------------------------------- #
# score normalisation
# --------------------------------------------------------------------------- #
@dataclass
class ScoreScale:
    """A 0..1 mapping of per-patch scores, plus everything a legend needs."""
    norm: np.ndarray                     # [N] in 0..1, the value fed to the cmap
    mode: str
    bar_label: str                       # colorbar axis label
    ticks: list = field(default_factory=list)   # [(pos01, label), ...]
    raw_min: float = 0.0
    raw_max: float = 0.0
    raw_median: float = 0.0
    clip_note: str = ""                  # e.g. "clipped at the 1st/99th pct"
    quantity: str = ATTENTION_SEMANTICS.quantity   # what the scores measure

    # measured at paint time, so the caption can describe what was drawn rather
    # than what the alpha ramp was hoped to do (see module docstring, point 4).
    render: str = "overlay"
    painted_frac: float = float("nan")   # fraction of the WHOLE image inked
    painted_frac_of_tissue: float = float("nan")   # ...of the patched tissue

    @property
    def raw_summary(self):
        return (f"raw {self.quantity} {self.raw_min:+.3g} to {self.raw_max:+.3g} "
                f"(median {self.raw_median:+.3g})")


def normalize_scores(scores, mode="rank", lo=1.0, hi=99.0,
                     quantity=ATTENTION_SEMANTICS.quantity):
    """Map raw per-patch attention to 0..1 for coloring.

    modes
    -----
    ``rank``       percentile of each tile among this slide's tiles. The default:
                   pre-softmax attention is bell-shaped, so this is the only
                   transform that reliably makes spatial structure visible.
                   Honest as long as the legend says "percentile" — it does.
    ``percentile`` linear in raw score, clipped to the [lo, hi] percentiles so a
                   couple of outlier tiles cannot flatten everything else.
                   Preserves relative magnitude; use when comparing pathways.
    ``minmax``     linear between the raw min and max (the original behaviour;
                   kept for reproducing older figures).
    ``signed``     symmetric about 0 at +/- the ``hi`` percentile of |score|, for
                   the diverging ramp: which tiles the pathway is aligned with
                   vs. anti-aligned with.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    good = np.isfinite(s)
    if not good.any():
        raise ValueError("attention vector has no finite values")
    s = np.where(good, s, np.nanmedian(s[good]))

    raw_min, raw_max = float(s.min()), float(s.max())
    raw_med = float(np.median(s))
    common = dict(raw_min=raw_min, raw_max=raw_max, raw_median=raw_med,
                  quantity=quantity)

    if mode == "rank":
        ranks = np.empty(s.size, dtype=np.float64)
        ranks[np.argsort(s, kind="stable")] = np.arange(s.size)
        norm = ranks / max(s.size - 1, 1)
        return ScoreScale(
            norm=norm, mode=mode,
            bar_label=f"{quantity} percentile among this slide's tiles",
            ticks=[(0.0, "0"), (0.25, "25th"), (0.5, "50th"),
                   (0.75, "75th"), (1.0, "100th")],
            clip_note="rank-normalised within this slide", **common)

    if mode == "signed":
        m = float(np.percentile(np.abs(s), hi))
        m = m if m > 0 else (abs(raw_max) or 1.0)
        norm = np.clip(0.5 + s / (2.0 * m), 0.0, 1.0)
        return ScoreScale(
            norm=norm, mode=mode,
            bar_label=f"{quantity} (signed)",
            ticks=[(0.0, f"{-m:+.3g}"), (0.25, f"{-m / 2:+.3g}"), (0.5, "0"),
                   (0.75, f"{m / 2:+.3g}"), (1.0, f"{m:+.3g}")],
            clip_note=f"symmetric about 0, saturating at +/-{m:.3g} "
                      f"(the {hi:g}th pct of |score|)", **common)

    if mode == "percentile":
        v0, v1 = np.percentile(s, [lo, hi])
        if v1 <= v0:
            v0, v1 = raw_min, raw_max
        norm = np.clip((s - v0) / ((v1 - v0) or 1e-8), 0.0, 1.0)
        return ScoreScale(
            norm=norm, mode=mode,
            bar_label=f"{quantity}",
            ticks=[(0.0, f"{v0:+.3g}"), (0.5, f"{(v0 + v1) / 2:+.3g}"),
                   (1.0, f"{v1:+.3g}")],
            clip_note=f"clipped to the {lo:g}th-{hi:g}th percentile "
                      f"({v0:+.3g} to {v1:+.3g})", **common)

    if mode == "minmax":
        norm = (s - raw_min) / ((raw_max - raw_min) or 1e-8)
        return ScoreScale(
            norm=norm, mode=mode,
            bar_label=f"{quantity}",
            ticks=[(0.0, f"{raw_min:+.3g}"),
                   (0.5, f"{(raw_min + raw_max) / 2:+.3g}"),
                   (1.0, f"{raw_max:+.3g}")],
            clip_note="linear between the raw min and max", **common)

    raise ValueError(f"unknown normalisation mode {mode!r}; expected one of "
                     f"rank / percentile / minmax / signed")


# --------------------------------------------------------------------------- #
# rasterise + composite
# --------------------------------------------------------------------------- #
def rasterize(shape_hw, coords, ds, values, stride_l0):
    """Paint per-patch ``values`` into a thumbnail-resolution raster.

    Returns ``(value_raster, covered_mask)``. Tile bounds are computed as
    ``round(coord / ds)`` .. ``round((coord + stride) / ds)`` rather than
    ``int(coord/ds) + int(stride/ds)``, so neighbouring tiles share an edge
    instead of leaving a sub-pixel seam. Overlapping tiles (CLAM step < size)
    keep the larger value.
    """
    H, W = shape_hw
    val = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=bool)

    xy = np.asarray(coords, dtype=np.float64)
    x0 = np.rint(xy[:, 0] / ds).astype(np.int64)
    y0 = np.rint(xy[:, 1] / ds).astype(np.int64)
    x1 = np.rint((xy[:, 0] + stride_l0) / ds).astype(np.int64)
    y1 = np.rint((xy[:, 1] + stride_l0) / ds).astype(np.int64)
    x1 = np.maximum(x1, x0 + 1)          # never smaller than one pixel
    y1 = np.maximum(y1, y0 + 1)

    x0 = np.clip(x0, 0, W); x1 = np.clip(x1, 0, W)
    y0 = np.clip(y0, 0, H); y1 = np.clip(y1, 0, H)

    v = np.asarray(values, dtype=np.float32).ravel()
    painted = 0
    for i in range(v.size):
        a, b, c, d = y0[i], y1[i], x0[i], x1[i]
        if b <= a or d <= c:
            continue
        tile = val[a:b, c:d]
        np.maximum(tile, v[i], out=tile)
        mask[a:b, c:d] = True
        painted += 1

    if painted < v.size:
        print(f"[warn] {v.size - painted} of {v.size} patches fell outside the "
              f"thumbnail and were not painted.")
    return val, mask


def smooth_raster(val, mask, sigma_px):
    """Gaussian-smooth the score raster without bleeding into unpatched slide.

    Normalised convolution: smooth value*mask and mask separately, then divide,
    so background pixels do not drag the tissue values toward zero.
    """
    if not sigma_px or sigma_px <= 0:
        return val
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("[warn] scipy not available -> skipping heatmap smoothing "
              "(tiles stay crisp; install scipy for a continuous field).")
        return val
    m = mask.astype(np.float32)
    num = gaussian_filter(val * m, sigma_px, mode="nearest")
    den = gaussian_filter(m, sigma_px, mode="nearest")
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), val)
    return out.astype(np.float32)


def composite(thumb, val, mask, cmap, alpha_max=0.85, alpha_min=0.0,
              alpha_gamma=1.5, focus_top=None, block=512, signed=False):
    """Blend the colored score raster onto the thumbnail.

    Alpha ramps with the score (``alpha_min`` -> ``alpha_max`` through
    ``norm ** alpha_gamma``) instead of being a flat 0.5. A flat alpha tints the
    entire slide and buries the signal in a wash; ramping it leaves cold tissue
    as clean H&E and spends ink only where the pathway actually attends, which
    is what makes the map read as specific.

    ``signed`` ramps alpha with *distance from the midpoint* instead, i.e.
    ``|2v - 1|``. On a diverging scale 0 and 1 are the two extremes and 0.5 is
    "nothing here", so the monotone ramp would fade the entire negative pole to
    invisible and leave a map that looks one-sided when it is not. Set this
    whenever the scale is zero-referenced (``norm_mode="signed"``).

    ``focus_top`` (e.g. 20) paints only the top N% of the score range — of
    ``|2v - 1|`` when ``signed``, so it keeps the strongest tiles of *both*
    poles — and leaves everything below it untouched.

    Row-blocked so a 3000x3000+ thumbnail never materialises a full float RGBA.
    """
    thumb = np.asarray(thumb)
    H, W = thumb.shape[:2]
    out = thumb[:, :, :3].astype(np.uint8).copy()

    floor = None
    if focus_top:
        floor = 1.0 - float(focus_top) / 100.0

    for r0 in range(0, H, block):
        r1 = min(r0 + block, H)
        m = mask[r0:r1]
        if not m.any():
            continue
        v = np.clip(val[r0:r1], 0.0, 1.0)
        # strength = what drives opacity: the score itself, or how far the score
        # sits from a diverging scale's neutral midpoint.
        strength = np.abs(2.0 * v - 1.0) if signed else v
        a = alpha_min + (alpha_max - alpha_min) * np.power(strength, alpha_gamma)
        a = np.where(m, a, 0.0).astype(np.float32)
        if floor is not None:
            a = np.where(strength >= floor, a, 0.0).astype(np.float32)
        if not (a > 0).any():
            continue
        rgb = (cmap(v)[:, :, :3] * 255.0).astype(np.float32)
        base = out[r0:r1].astype(np.float32)
        a3 = a[:, :, None]
        out[r0:r1] = np.clip(base * (1.0 - a3) + rgb * a3, 0, 255).astype(np.uint8)

    return out


def mosaic(shape_hw, val, mask, cmap, background=MOSAIC_BG, thumb=None,
           texture=0.0, block=512):
    """Draw the score raster as opaque tiles on a flat canvas — the paper look.

    This is the A_{P->H} panel from the SurvPath figures: a tissue-shaped mosaic
    floating on white, with nothing of the H&E showing through. Unlike
    ``composite`` it makes no claim about where the model did or did not look —
    every patched tile is drawn at full strength, and the only thing left
    unpainted is slide with no patch on it (background CLAM excluded). That
    removes the contradiction the alpha-ramped overlay had, where "unpainted =
    barely looked" was printed under an image that had inked all the tissue.

    ``texture`` (0..1) multiplies each tile by the local H&E luminance, which
    brings a hint of tissue architecture back without reintroducing the
    everything-is-tinted problem. 0 = flat color, as published.
    """
    from matplotlib.colors import to_rgb

    H, W = shape_hw
    bg = np.array([round(c * 255) for c in to_rgb(background)], dtype=np.uint8)
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[:, :] = bg

    lum = None
    if texture > 0 and thumb is not None:
        t = np.asarray(thumb)[:, :, :3].astype(np.float32)
        lum = (0.299 * t[:, :, 0] + 0.587 * t[:, :, 1] + 0.114 * t[:, :, 2]) / 255.0

    for r0 in range(0, H, block):
        r1 = min(r0 + block, H)
        m = mask[r0:r1]
        if not m.any():
            continue
        v = np.clip(val[r0:r1], 0.0, 1.0)
        rgb = cmap(v)[:, :, :3] * 255.0
        if lum is not None:
            shade = (1.0 - texture) + texture * lum[r0:r1]
            rgb = rgb * shade[:, :, None]
        tile = out[r0:r1]
        tile[m] = np.clip(rgb, 0, 255).astype(np.uint8)[m]

    return out


def paint_attention(thumb, coords, ds, scores, stride_l0, cmap_name=DEFAULT_CMAP,
                    norm_mode="rank", pct_lo=1.0, pct_hi=99.0,
                    smooth_tiles=0.5, alpha_max=0.85, alpha_min=0.0,
                    alpha_gamma=1.5, focus_top=None, render="overlay",
                    background=MOSAIC_BG, texture=0.0, warn_cmap=True,
                    semantics=ATTENTION_SEMANTICS):
    """Full pipeline for one pathway: scores -> painted image + its scale.

    ``render``
        ``"overlay"``  blend onto the H&E with alpha ramped by score.
        ``"mosaic"``   opaque tiles on a flat canvas (the published A_P->H look).

    ``smooth_tiles`` is the Gaussian sigma in units of *tile widths*, so the
    amount of smoothing is independent of the thumbnail downsample.

    The returned ``ScoreScale`` carries how much of the image actually got ink,
    measured from the result — captions quote that rather than assuming.
    """
    coords = np.asarray(coords)
    scores = np.asarray(scores).ravel()
    if coords.shape[0] != scores.shape[0]:
        raise ValueError(
            f"coords ({coords.shape[0]}) != attention patches ({scores.shape[0]}). "
            f"The .h5 must come from the same CLAM run that produced the .pt "
            f"features (same patch_size/step, no subsampling).")

    scale = normalize_scores(scores, mode=norm_mode, lo=pct_lo, hi=pct_hi,
                             quantity=semantics.quantity)
    cmap = resolve_cmap(cmap_name, warn=warn_cmap)

    val, mask = rasterize(thumb.shape[:2], coords, ds, scale.norm, stride_l0)
    tile_px = max(stride_l0 / ds, 1.0)
    val = smooth_raster(val, mask, sigma_px=smooth_tiles * tile_px)

    if render == "mosaic":
        img = mosaic(thumb.shape[:2], val, mask, cmap, background=background,
                     thumb=thumb, texture=texture)
        inked = mask
    elif render == "overlay":
        img = composite(thumb, val, mask, cmap, alpha_max=alpha_max,
                        alpha_min=alpha_min, alpha_gamma=alpha_gamma,
                        focus_top=focus_top, signed=(scale.mode == "signed"))
        # what changed is what got ink — measured, not assumed
        inked = (img != np.asarray(thumb)[:, :, :3]).any(axis=-1)
    else:
        raise ValueError(f"unknown render mode {render!r}; expected "
                         f"'overlay' or 'mosaic'")

    scale.render = render
    scale.painted_frac = float(inked.mean())
    tissue = int(mask.sum())
    scale.painted_frac_of_tissue = (
        float(inked[mask].mean()) if tissue else float("nan"))
    return img, scale


# --------------------------------------------------------------------------- #
# legend + caption
# --------------------------------------------------------------------------- #
def color_caption(scale, cmap_name=DEFAULT_CMAP, stride_l0=None, ds=None,
                  subject="this pathway", focus_top=None, width=112,
                  diagnostics=None, specificity=None,
                  semantics=ATTENTION_SEMANTICS):
    """Spell out, in words, what the reader is actually looking at.

    A colorbar alone says "darker = more" but not more *of what*, says nothing
    about the tissue left unpainted, and — worst of all — says nothing about
    whether the pattern is real. So this text is generated from measurements
    taken during painting (`scale.painted_frac_of_tissue`, `scale.render`) plus,
    when the caller supplies them, the Moran's I and cross-pathway correlation
    reports. Nothing here is a standing assumption about how the map "should"
    have come out.
    """
    lines = []
    tile = ""
    if stride_l0:
        tile = f"{stride_l0} px tile"
        if ds:
            tile += f" (~{stride_l0 / ds:.0f} px on this thumbnail)"
        tile = f" per {tile}"

    words = dict(subject=subject, tile=tile,
                 lo=_hex_at(cmap_name, 0.05), mid=_hex_at(cmap_name, 0.5),
                 hi=_hex_at(cmap_name, 0.95))

    lines.append(f"COLOR = {semantics.lead.format(**words)}, "
                 f"{scale.clip_note}.")

    if cmap_name in RAINBOW_CMAPS:
        lines.append(
            f"Ramp order is blue -> cyan -> green -> yellow -> red (low to "
            f"high), matching the published figure. Note this ramp's lightness "
            f"is not monotone: use it to spot *where* the extremes are, and the "
            f"colorbar to read a value — two mid-range tiles cannot be reliably "
            f"ranked by eye.")
    elif is_diverging(cmap_name):
        lines.append(semantics.diverging.format(**words))
    else:
        lines.append(semantics.sequential.format(**words))

    lines.append(_coverage_sentence(scale, focus_top, semantics))

    if diagnostics is not None:
        lines.append(diagnostics.summary())
    if specificity is not None and specificity.summary():
        lines.append(specificity.summary())

    lines.append(f"Scale is per-slide and not comparable across cases: "
                 f"{scale.raw_summary}.")

    return "\n".join(textwrap.fill(ln, width=width) for ln in lines)


def _coverage_sentence(scale, focus_top=None, semantics=ATTENTION_SEMANTICS):
    """Describe what got ink, from the measurement rather than the intent."""
    if scale.render == "mosaic":
        return ("Every extracted patch is drawn at full opacity, so this is a "
                "ranking across all of the tissue, not a highlight of a few "
                "regions; blank canvas means no patch was extracted there "
                "(background / excluded by CLAM), not low attention.")

    frac = scale.painted_frac_of_tissue
    if not np.isfinite(frac):
        return ("Uncovered slide means no patch was extracted there "
                "(background / excluded by CLAM).")

    if focus_top:
        what = ("of the range by distance from zero, so both directions are kept"
                if scale.mode == "signed" else "of the range")
        return (f"Only the top {focus_top:g}% {what} is painted — "
                f"{frac:.0%} of the patched tissue took ink; the rest is left as "
                f"plain H&E.")

    if frac >= 0.9:
        return (f"Alpha ramps with the score, but {frac:.0%} of the patched "
                f"tissue still took ink: this is a ranking spread over the whole "
                f"section, NOT a few localised hotspots. Read only the most "
                f"saturated tiles as '{semantics.hot_word}'; re-run with "
                f"--focus-top 10 to paint just those.")
    return (f"Alpha ramps with the score and {frac:.0%} of the patched tissue "
            f"took ink, so paler / unpainted H&E is {semantics.low_phrase}. "
            f"Uncovered slide means no patch was extracted there "
            f"(background / excluded by CLAM).")


def add_colorbar(fig, rect, scale, cmap_name=DEFAULT_CMAP):
    """Horizontal colorbar with the scale's own ticks, in muted ink."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    cax = fig.add_axes(rect)
    sm = ScalarMappable(norm=Normalize(0.0, 1.0),
                        cmap=resolve_cmap(cmap_name, warn=False))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([p for p, _ in scale.ticks])
    cb.set_ticklabels([lab for _, lab in scale.ticks])
    cb.set_label(scale.bar_label, color=INK_SECONDARY, fontsize=9)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8, length=2, width=0.6)
    cb.outline.set_edgecolor(HAIRLINE)
    cb.outline.set_linewidth(0.6)
    return cb


def save_overlay_figure(overlay, scale, out_path, title, caption,
                        cmap_name=DEFAULT_CMAP, subtitle=None, width_in=11.0,
                        dpi=200):
    """One overlay + colorbar + caption, sized to the thumbnail's aspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = overlay.shape[:2]
    img_h = width_in * (H / max(W, 1))
    n_cap = caption.count("\n") + 1
    cap_h = 0.19 * n_cap + 0.25          # caption block
    bar_h = 0.75                          # colorbar + its label
    head_h = 0.60 + (0.26 if subtitle else 0.0)
    fig_h = img_h + cap_h + bar_h + head_h

    fig = plt.figure(figsize=(width_in, fig_h), facecolor=SURFACE)
    img_bottom = (cap_h + bar_h) / fig_h
    ax = fig.add_axes([0.02, img_bottom, 0.96, img_h / fig_h])
    ax.imshow(overlay, interpolation="nearest")
    ax.set_axis_off()

    ax.set_title(title, color=INK, fontsize=13, pad=10 if subtitle else 6)
    if subtitle:
        fig.text(0.5, img_bottom + img_h / fig_h + 0.20 / fig_h, subtitle,
                 ha="center", va="bottom", color=INK_SECONDARY, fontsize=9.5)

    add_colorbar(fig, [0.22, (cap_h + 0.42) / fig_h, 0.56, 0.16 / fig_h],
                 scale, cmap_name)
    fig.text(0.5, (cap_h - 0.14) / fig_h, caption, ha="center", va="top",
             color=INK_SECONDARY, fontsize=8.5, linespacing=1.5)

    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
