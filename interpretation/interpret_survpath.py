"""
Interpretability for SurvPath — three complementary views:

  (1) Captum Integrated Gradients  -> WHICH pathways / genes drive the risk
      prediction for a case, and BY HOW MUCH. The IG target is the model's own
      risk score, so <case>_pathway_importance.csv is in risk units: a
      ``risk_delta`` of +0.4 means that pathway's expression added 0.4 to this
      case's predicted risk (``pct_of_risk_change`` says how big a share of the
      whole move that was), and a negative one lowered the risk.

  (2) Pathway -> patch cross-attention -> WHERE on the WSI a given pathway
      "looks". This is the raw material for the spatial heatmaps in the paper.

  (2b) Integrated Gradients on the patch features -> WHICH TISSUE moved the
      risk prediction, and in which direction. (2) is a routing weight and can
      be large on tissue that changes nothing; this is a gradient of the risk
      output itself, so red means "raised this case's risk" and blue means
      "lowered it". Written as <case>_ig_heatmap.png.

    INTERP_CKPT        path to a trained checkpoint, e.g.
                       results/<exp>/s_0_checkpoint.pt          (REQUIRED)
    INTERP_FOLD        which fold's val split to read a case from   (default 0)
    INTERP_CASE_ID     case_id to interpret, e.g. TCGA-AN-A0FF (a unique
                       substring like A0FF also works). PREFER THIS over
                       INTERP_CASE_IDX, and it wins if both are set.
    INTERP_CASE_IDX    index of the case within the val split       (default 0)
                       NB: this is a position in the dataset as constructed, NOT
                       row N of splits_<fold>.csv. Cases whose .pt features are
                       missing from data_root_dir are dropped first, so every
                       later index shifts down by one per missing case and moves
                       again whenever the feature directory changes; rows are
                       slide-level, so multi-slide cases take two indices.
    INTERP_OUTDIR      where to write outputs           (default ./interpret_out)
    INTERP_TOPK        how many top pathways to visualise           (default 10)
    INTERP_IG_STEPS    Integrated-Gradients steps                   (default 50)
    INTERP_IG_BATCH    IG steps evaluated at once; lower it if the WSI bag
                       OOMs, raise it to go faster       (default 8)
    INTERP_IG_REDUCE   how to collapse the per-feature attributions into one
                       number per patch: signed_sum (default, keeps IG's
                       completeness and its sign) | abs_sum | l2
    INTERP_IG_NORM     normalisation for the IG map alone
                       (default: signed when the reduction is signed)
    INTERP_IG_CMAP     colormap for the IG map alone
                       (default: attention-signed when signed)

  Spatial overlay :
    INTERP_COORDS_DIR  dir of CLAM patch files "<slide_id>.h5" holding a
                       'coords' dataset (level-0 pixel x,y per patch)
    INTERP_SLIDE_DIR   dir of the whole-slide images ("<slide_id>.svs" etc.)
    INTERP_PATCH_SIZE  patch edge in level-0 pixels                 (default 256)

If COORDS_DIR / SLIDE_DIR are not given, the script still writes the per-patch
attention vectors (.npy) and the Captum rankings (.csv); it just skips painting
them onto the slide and prints how to regenerate the coordinates.

To Run:
    INTERP_CKPT=results/tcga_brca__survpath/s_0_checkpoint.pt \
    INTERP_OUTDIR=results/interpret \
    INTERP_TOPK=10 \
    python -m interpretation.interpret_survpath \
        --study tcga_brca --modality survpath --type_of_path combine \
        --data_root_dir /path/to/pt_features  ... (rest as in survpath.sh)
"""

import json
import os
import numpy as np
if not hasattr(np, "typeDict"):          # removed in NumPy 1.24; h5py 2.x still uses it
    np.typeDict = np.sctypeDict
import pandas as pd
import torch

from datasets.dataset_survival import SurvivalDatasetFactory
from utils.process_args import _process_args
from utils.general_utils import _prepare_for_experiment
from models.model_SurvPath import SurvPath

# Coords/slide lookup + the up-front file check live in heatmap_utils so that
# render_gene_pathway_map.py resolves these files identically.
from interpretation.heatmap_utils import check_case_files, IG_SEMANTICS
# The spatial overlay itself (canvas + painting) is its own module so it can be
# imported without dragging in the model, the dataset factory or Captum.
from interpretation.render_overlay import (
    render_overlay,
    slide_canvas as _slide_canvas,
    paint_and_save as _paint_and_save,
)
from interpretation.pathway_names import PathwayNames


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _env(name, default=None, cast=str):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


def _heatmap_style():
    """Heatmap look-and-feel, all optional (see heatmap_utils for the defaults).

        INTERP_CMAP        'attention' (sequential blue, default) |
                           'attention-signed' (diverging, needs NORM=signed) |
                           any matplotlib name
        INTERP_NORM        rank (default) | percentile | minmax | signed
        INTERP_PCT_LO/HI   clip percentiles for NORM=percentile   (1 / 99)
        INTERP_SMOOTH      Gaussian sigma in tile widths          (0.5)
        INTERP_ALPHA_MAX   opacity of the hottest tissue          (0.85)
        INTERP_ALPHA_MIN   opacity of the coldest tissue          (0.0)
        INTERP_ALPHA_GAMMA >1 concentrates ink on the hot tail    (1.5)
        INTERP_FOCUS_TOP   paint only the top N% of the range     (unset = all)
        INTERP_DOWNSAMPLE  thumbnail downsample                   (32)
        INTERP_RENDER      overlay (default) | mosaic — 'mosaic' is the
                           published A_P->H look: opaque tiles on a flat canvas
                           with no H&E showing through
        INTERP_TEXTURE     0..1, shade mosaic tiles by H&E luminance    (0)
    """
    style = {
        "cmap_name": _env("INTERP_CMAP", "attention"),
        "norm_mode": _env("INTERP_NORM", "rank"),
        "pct_lo": _env("INTERP_PCT_LO", 1.0, float),
        "pct_hi": _env("INTERP_PCT_HI", 99.0, float),
        "smooth_tiles": _env("INTERP_SMOOTH", 0.5, float),
        "alpha_max": _env("INTERP_ALPHA_MAX", 0.85, float),
        "alpha_min": _env("INTERP_ALPHA_MIN", 0.0, float),
        "alpha_gamma": _env("INTERP_ALPHA_GAMMA", 1.5, float),
        "focus_top": _env("INTERP_FOCUS_TOP", None, float),
        "downsample": _env("INTERP_DOWNSAMPLE", 32, float),
        "render": _env("INTERP_RENDER", "overlay"),
        "texture": _env("INTERP_TEXTURE", 0.0, float),
    }
    if style["norm_mode"] == "signed" and style["cmap_name"] == "attention":
        style["cmap_name"] = "attention-signed"   # signed scores need two poles
    return style


def _build_args():
    """Reproduce main.py's arg + dataset-factory construction exactly."""
    args = _process_args()
    args = _prepare_for_experiment(args)
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed,
        print_info=True,
        n_bins=args.n_classes,
        label_col=args.label_col,
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat="coattn" in args.modality,
        is_survpath=args.modality == "survpath",
        type_of_pathway=args.type_of_path,
    )
    return args


def _load_model(ckpt_path, args, omic_names, device):
    """Instantiate SurvPath to match the checkpoint and load weights.

    ``wsi_embedding_dim`` is read straight from the checkpoint's projection
    layer so this works regardless of which patch encoder produced the .pt
    features (ResNet-50 = 1024, UNI = 1024, CONCH = 512, ...).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    wsi_dim = state["wsi_projection_net.0.weight"].shape[1]

    model = SurvPath(
        omic_sizes=args.omic_sizes,
        num_classes=args.n_classes,
        wsi_embedding_dim=wsi_dim,
        omic_names=omic_names,      # needed so model.all_gene_names exists
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return model


def _fetch_case(val_split, idx, device):
    """Pull one val-split case. val split uses sample=False, so patches keep
    their on-disk order — this is what lets us line them up with .h5 coords.

    ``event_time`` / ``censorship`` come back too: the published figures caption
    each case with "Last follow-up: 33 months" (censored) or "Survival time: 18
    months" (an observed death), and that distinction is exactly the censorship
    flag.
    """
    patch_features, omic_list, label, event_time, c, clinical, mask = val_split[idx]
    # slide ids / case id for locating the CLAM .h5 and the WSI later
    _, _, _, slide_ids, _, case_id = val_split.get_data_to_return(idx)

    wsi = patch_features.unsqueeze(0).float().to(device)          # [1, N, D]
    omics = [o.float().to(device) for o in omic_list]             # P x [size_p]
    return wsi, omics, slide_ids, case_id, float(event_time), float(c)


def _risk_from_logits(logits):
    """The model's own risk definition, reproduced from the logits.

    Kept identical to ``models/model_SurvPath.py:197`` and
    ``utils/core_utils._calculate_risk``: discrete-time hazards -> survival ->
    risk = -sum(S). Recomputed here rather than re-running the model, because
    ``extract_attention`` already has the logits.
    """
    h = torch.as_tensor(logits, dtype=torch.float32)
    survival = torch.cumprod(1 - torch.sigmoid(h), dim=-1)
    return float(-torch.sum(survival, dim=-1).ravel()[0])


# --------------------------------------------------------------------------- #
# (1) attention: pathway -> patch
# --------------------------------------------------------------------------- #
def extract_attention(model, wsi, omics):
    """Returns cross_attn_pathways [P, N]: row p = pathway p's attention over
    the N patches (raw, pre-softmax scores)."""
    input_args = {"x_path": wsi, "return_attn": True}
    for i, o in enumerate(omics):
        input_args[f"x_omic{i + 1}"] = o
    with torch.no_grad():
        logits, attn_pathways, cross_attn_pathways, cross_attn_histology = model(**input_args)
    # squeeze()'d in the layer to [P, N] (single head, batch 1)
    return cross_attn_pathways.float().numpy(), logits.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# (2) Captum: pathway / gene importance
# --------------------------------------------------------------------------- #
def captum_importance(model, wsi, omics, omic_names, n_steps, device,
                      pathway_names=None, internal_batch_size=8):
    """Integrated gradients of the risk score w.r.t. every input.

    Returns ``(path_df, gene_df, wsi_attr, delta)``. ``wsi_attr`` is
    ``[1, N, D]`` — the attribution of each patch's encoder features, which is
    what the spatial IG heatmap is made of; it used to be computed and thrown
    away. ``delta`` is IG's convergence error (below).

    ``internal_batch_size`` chunks the ``n_steps`` Riemann samples. Left unset,
    Captum expands every input by ``n_steps`` at once: a 4096-patch bag at
    1024-d is ~16 MB, so 50 steps is ~800 MB of activations before gradients
    and 331 pathway tensors on top. Chunking costs wall-clock, not accuracy.
    """
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model.captum)
    # model.captum(self, omics_0..omics_330, wsi) -> risk. Give each omic a
    # batch dim; wsi is last.
    inputs = tuple(o.unsqueeze(0) for o in omics) + (wsi,)
    # All-zero baseline. This is Captum's default, but for a heatmap it is part
    # of the claim being made — every number below is "relative to a case with
    # no expression and no tissue" — so it is stated rather than inherited.
    baselines = tuple(torch.zeros_like(t) for t in inputs)
    attrs, delta = ig.attribute(inputs, baselines=baselines, n_steps=n_steps,
                                internal_batch_size=internal_batch_size,
                                return_convergence_delta=True)
    delta = float(torch.as_tensor(delta).abs().max())

    path_rows, gene_rows = [], []
    for p, a in enumerate(attrs[:-1]):                 # last entry is the WSI
        a = a.detach().cpu().numpy().ravel()           # [size_p]
        names = list(omic_names[p])
        net, mag = float(a.sum()), float(np.abs(a).sum())
        path_rows.append({"pathway_idx": p,
                          "pathway": _pathway_label(pathway_names, p),
                          # How much this pathway moved the risk, in risk units:
                          # model.captum returns risk = -sum(S) directly
                          # (model_SurvPath.py:199), so IG attributes that scalar
                          # and nothing here rescales it. Positive = pushed this
                          # case toward the worse prognosis.
                          "risk_delta": net,
                          "direction": "increases risk" if net > 0
                                       else "decreases risk" if net < 0 else "none",
                          # Total gene-level movement, ignoring direction. It and
                          # risk_delta answer different questions and can disagree
                          # sharply, so the disagreement is a column rather than
                          # something you have to notice. `coherence` =
                          # |net| / magnitude is how much of the pathway's
                          # gene-level influence survived cancellation: a pathway
                          # whose genes push +5/-5 has a large abs_risk_delta,
                          # coherence ~0, and no effect on the prediction —
                          # ranking it "top" by magnitude alone is the trap.
                          "abs_risk_delta": mag,
                          "coherence": abs(net) / mag if mag > 0 else 0.0,
                          "n_genes": len(names)})
        for g, name in enumerate(names):
            if g < a.shape[0]:
                gene_rows.append({"pathway_idx": p, "gene": name,
                                  "attr": float(a[g]), "abs_attr": float(abs(a[g]))})

    wsi_attr = attrs[-1].detach().cpu().numpy()        # [1, N, D]
    path_df = _finish_pathway_table(pd.DataFrame(path_rows),
                                    wsi_net=float(wsi_attr.sum()))
    gene_df = pd.DataFrame(gene_rows).sort_values("abs_attr", ascending=False)
    return path_df, gene_df, wsi_attr, delta


def _finish_pathway_table(path_df, wsi_net):
    """Add the share-of-risk column, rank, and order the rows for reading.

    ``risk_delta`` on its own says +0.4 without saying +0.4 out of what, so the
    file cannot be read without also opening summary.json. IG's completeness
    axiom supplies the denominator: the pathway attributions plus the WSI
    attribution sum to ``risk(case) - risk(baseline)``, so a pathway's signed
    fraction of that total is a statement you can actually make — "this pathway
    accounts for 12% of why this case scored badly".

    The share is taken against the net move, so a pathway pushing the same way
    the prediction went reads positive whichever way that was; ``direction`` is
    the absolute statement. Rows are sorted signed, biggest risk increase first
    and biggest decrease last, and ``rank_by_risk_delta`` carries the |effect|
    ordering that the heatmaps key off so both readings are in the file.
    """
    total = float(path_df["risk_delta"].sum()) + wsi_net
    gross = float(path_df["risk_delta"].abs().sum()) + abs(wsi_net)
    if abs(total) <= 1e-6 * max(gross, 1e-12):
        # Dividing by this would turn rounding error into confident percentages.
        path_df["pct_of_risk_change"] = np.nan
        print("[warn] the pathway and WSI attributions net out to ~0, so "
              "'share of the risk change' has no denominator — "
              "pct_of_risk_change left blank; read risk_delta (risk units).")
    else:
        path_df["pct_of_risk_change"] = 100.0 * path_df["risk_delta"] / total
        if abs(total) < 0.1 * gross:
            print(f"[note] contributions largely cancel (net {total:+.3g} vs "
                  f"gross {gross:.3g}), so single pct_of_risk_change values can "
                  f"exceed 100%.")
    path_df["rank_by_risk_delta"] = (path_df["risk_delta"].abs()
                                     .rank(ascending=False, method="min")
                                     .astype(int))
    cols = ["pathway_idx", "pathway", "risk_delta", "direction",
            "pct_of_risk_change", "rank_by_risk_delta", "abs_risk_delta",
            "coherence", "n_genes"]
    return path_df[cols].sort_values("risk_delta", ascending=False)


def print_direction_report(path_df, topk):
    """Which pathways pushed this case's risk up, and which pushed it down.

    ``risk = -sum(S)`` (``_risk_from_logits``), so it rises as predicted survival
    falls: a POSITIVE attribution moved this case toward the worse prognosis, a
    negative one toward the better. IG's completeness makes these net numbers
    additive — the positive and negative columns below sum to the omics half of
    ``risk(case) - risk(baseline)`` — which is why the split is reported on
    ``risk_delta`` and not on ``abs_risk_delta``.
    """
    up = path_df[path_df["risk_delta"] > 0].sort_values("risk_delta", ascending=False)
    down = path_df[path_df["risk_delta"] < 0].sort_values("risk_delta")

    for label, df in (("RAISED risk", up), ("LOWERED risk", down)):
        total = float(df["risk_delta"].sum())
        print(f"\n  Pathways that {label} ({len(df)} of {len(path_df)}, "
              f"net {total:+.3g} risk):")
        for r in df.head(topk).itertuples(index=False):
            # Flag the rows where the magnitude ranking and the net effect part
            # ways, i.e. mostly-cancelling pathways riding high on |IG|.
            note = "  (genes largely cancel)" if r.coherence < 0.25 else ""
            share = ("" if not np.isfinite(r.pct_of_risk_change)
                     else f" ({r.pct_of_risk_change:+.1f}% of the change)")
            print(f"    {r.risk_delta:+9.3g}  [{r.pathway_idx:>3}] "
                  f"{r.pathway}{share}{note}")

    # Magnitude ranks by total gene-level movement and so cannot distinguish
    # "strong driver" from "internally cancelling"; say so where the two lists
    # actually differ rather than as a general caveat.
    by_mag = set(path_df.nlargest(topk, "abs_risk_delta")["pathway_idx"])
    by_net = set(path_df.reindex(
        path_df["risk_delta"].abs().nlargest(topk).index)["pathway_idx"])
    if by_mag - by_net:
        names = ", ".join(
            str(path_df.loc[path_df["pathway_idx"] == p, "pathway"].iloc[0])
            for p in sorted(by_mag - by_net))
        print(f"\n  [note] top-{topk} by |IG| but not by net effect (their genes "
              f"cancel): {names}")


def _resolve_case_idx(val_split, wanted):
    """Position of ``wanted`` in this fold's val dataset, or a useful error.

    ``INTERP_CASE_IDX`` is a position in the *constructed* dataset, which is not
    row N of ``splits_<fold>.csv``: ``_filter_split_to_available_wsi``
    (dataset_survival.py) first drops every case with no .pt on disk, so each
    missing feature file shifts all later indices down by one, and the drop list
    changes whenever the feature directory does. The rows are also slide-level,
    so a two-slide case occupies two consecutive indices. Naming the case is the
    only way to address it that survives both.

    Matching is exact on case_id, then falls back to a substring so
    ``INTERP_CASE_ID=A0FF`` works; an ambiguous substring is an error rather
    than a silent first-hit.
    """
    ids = val_split.metadata["case_id"].astype(str)
    wanted = wanted.strip()

    hits = ids.index[ids == wanted].tolist()
    how = "exact"
    if not hits:
        hits = ids.index[ids.str.contains(wanted, case=False, regex=False)].tolist()
        how = "substring"
        matched = sorted(set(ids[hits]))
        if len(matched) > 1:
            shown = ", ".join(matched[:8])
            more = f", ... (+{len(matched) - 8} more)" if len(matched) > 8 else ""
            raise SystemExit(
                f"INTERP_CASE_ID={wanted!r} matches {len(matched)} cases in fold "
                f"{val_split.fold}'s val split: {shown}{more}\n"
                f"       Give the full case_id.")
    if not hits:
        raise SystemExit(
            f"INTERP_CASE_ID={wanted!r} is not in fold {val_split.fold}'s val "
            f"split ({len(set(ids))} cases with WSI features on disk).\n"
            f"       It may be in the train split, in another fold, or dropped "
            f"for having no .pt file - see the 'WARNING: skipping N case(s)' "
            f"line above.")

    idx = hits[0]
    note = "" if how == "exact" else f" (substring match on {ids[idx]})"
    extra = (f"; case spans {len(hits)} slide-level rows {hits}, using the first"
             if len(hits) > 1 else "")
    print(f"[info] INTERP_CASE_ID={wanted} -> val_split index {idx}{note}{extra}")
    return idx


def _pathway_label(pathway_names, p):
    """The signature CSV's column name for token p, or the old placeholder.

    Pathway tokens are the signature CSV's columns in file order
    (datasets/dataset_survival.py:147-158), so this is a positional lookup. It
    used to return ``pathway_<p>`` unconditionally, which is why the figures
    were titled with a bare integer instead of "Epithelial Mesenchymal
    Transition".
    """
    if pathway_names:
        return pathway_names.raw(p)
    return f"pathway_{p}"


# --------------------------------------------------------------------------- #
# (2b) integrated gradients as a spatial map
#
# The cross-attention heatmap answers "where does pathway p look?". It says
# nothing about the prediction: attention is a routing weight, and a tile can be
# attended to hard while contributing nothing to the risk. IG answers the other
# question — "which tissue moved this case's predicted risk, and which way?" —
# because it is a gradient of the risk output itself, w.r.t. the patch features.
#
# Captum hands back one attribution per feature, [1, N, D]. Collapsing D is a
# choice with consequences, so it is explicit (INTERP_IG_REDUCE):
#
#   signed_sum  (default)  sum over D. Keeps IG's completeness property: these
#                          per-patch numbers add up (with the pathway
#                          attributions) to risk(case) - risk(baseline), so
#                          "this region accounts for +0.3 of the risk" is a
#                          statement you can actually make. Signed -> pair with
#                          the diverging ramp.
#   abs_sum                sum of |attribution|. Total influence regardless of
#                          direction; loses completeness and cancellation, so a
#                          tile pushing +5/-5 on two features reads as hot.
#   l2                     Euclidean norm over D. Same "magnitude only" reading
#                          as abs_sum but dominated by single large features.
# --------------------------------------------------------------------------- #
IG_REDUCERS = {
    "signed_sum": lambda a: a.sum(axis=-1),
    "abs_sum": lambda a: np.abs(a).sum(axis=-1),
    "l2": lambda a: np.sqrt((a ** 2).sum(axis=-1)),
}


def patch_attribution(wsi_attr, reduce="signed_sum"):
    """[1, N, D] IG attributions -> [N] per-patch score.

    Returns ``(values, is_signed)``; ``is_signed`` tells the caller whether zero
    is meaningful, which decides the colormap and the alpha ramp.
    """
    if reduce not in IG_REDUCERS:
        raise ValueError(f"unknown INTERP_IG_REDUCE {reduce!r}; expected one of "
                         f"{'/'.join(IG_REDUCERS)}")
    a = np.asarray(wsi_attr, dtype=np.float64)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError(f"expected a single case, got batch {a.shape[0]}")
        a = a[0]                                       # [N, D]
    if a.ndim != 2:
        raise ValueError(f"expected [1, N, D] or [N, D] attributions, got {a.shape}")
    return IG_REDUCERS[reduce](a), reduce == "signed_sum"


def ig_style(base_style, is_signed):
    """Style for the IG map, defaulting to the reading its values support.

    A signed, zero-referenced quantity needs a diverging ramp about zero — the
    sequential/rank default inherited from the attention maps would colour the
    most protective tile and the most neutral tile the same pale blue and put
    the risk-driving tiles and the risk-lowering tiles at opposite ends of a
    ramp captioned "more". Explicit INTERP_IG_* settings still win.
    """
    from interpretation import heatmap_utils as hm

    style = dict(base_style)
    if is_signed:
        style["norm_mode"] = _env("INTERP_IG_NORM", "signed")
        style["cmap_name"] = _env("INTERP_IG_CMAP", hm.DEFAULT_SIGNED_CMAP)
    else:
        style["norm_mode"] = _env("INTERP_IG_NORM", base_style.get("norm_mode", "rank"))
        style["cmap_name"] = _env("INTERP_IG_CMAP", hm.DEFAULT_CMAP)
    return style


# --------------------------------------------------------------------------- #
# (3) spatial overlay (optional)
# --------------------------------------------------------------------------- #
def _slide_canvas(slide_id, coords_dir, slide_dir, downsample, n_patches,
                  patch_size):
    """Load the coords + thumbnail every per-patch map is painted on.

    Shared by the attention and the integrated-gradients renderers so both
    resolve files, choose a pyramid level and infer the tile stride identically
    — a map painted on a different grid than the one beside it is not
    comparable, and that is the whole point of putting them in one folder.

    Returns ``(coords, thumb, ds, stride)`` or ``None`` if anything is missing.
    """
    import h5py
    import openslide
    from interpretation import heatmap_utils as hm

    h5_path = _find(coords_dir, slide_id, (".h5",))
    slide_path = _find(slide_dir, slide_id, _SLIDE_EXTS)
    if h5_path is None or slide_path is None:
        print(f"[skip] no coords/slide for {slide_id} "
              f"(h5={h5_path}, slide={slide_path})")
        return None

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]                        # [N, 2] level-0 px
    if coords.shape[0] != n_patches:
        print(f"[skip] {slide_id}: coords ({coords.shape[0]}) != patches "
              f"({n_patches}). Multi-slide case or subsampling mismatch.")
        return None

    slide = openslide.OpenSlide(slide_path)
    # pick a downsample that keeps the thumbnail manageable
    level = slide.get_best_level_for_downsample(downsample)
    ds = slide.level_downsamples[level]
    thumb = np.array(
        slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))

    # Tiles must cover the grid the features were extracted on, not an assumed
    # 256 px: a wrong footprint is what turned these maps into isolated dots.
    stride = hm.infer_patch_stride(coords, declared=patch_size)
    return coords, thumb, ds, stride


def _paint_and_save(values, coords, thumb, ds, stride, out_path, title, subject,
                    style, semantics, label=""):
    """Paint one per-patch vector and write the annotated figure."""
    from interpretation import heatmap_utils as hm

    overlay, scale = hm.paint_attention(thumb, coords, ds, values, stride,
                                        semantics=semantics, **style)

    # Rank normalisation makes any input look vivid, so test whether this map is
    # spatially structured at all and put the answer in the caption.
    diag = hm.spatial_diagnostics(coords, values, stride)
    print(f"[info] {label or subject}: {diag.summary()}")

    cmap_name = style.get("cmap_name", hm.DEFAULT_CMAP)
    caption = hm.color_caption(scale, cmap_name=cmap_name, stride_l0=stride,
                               ds=ds, subject=subject,
                               focus_top=style.get("focus_top"),
                               diagnostics=diag, semantics=semantics)
    hm.save_overlay_figure(
        overlay, scale, out_path, title=title, caption=caption,
        cmap_name=cmap_name,
        subtitle=f"{coords.shape[0]} tiles · {stride} px at level 0 · "
                 f"thumbnail downsample {ds:.0f}x")
    print(f"[ok] wrote {out_path}")
    return True


def render_overlay(attn_vec, slide_id, coords_dir, slide_dir, patch_size,
                   out_path, pathway_idx=None, style=None):
    """Paint one pathway's per-patch attention onto a slide thumbnail.

    The painting itself (tile footprint, normalisation, ramp, alpha, legend,
    caption) lives in ``interpretation.heatmap_utils`` so this and
    ``render_gene_pathway_map.py`` produce identical, self-describing figures.
    """
    style = dict(style or {})
    downsample = style.pop("downsample", 32)

    canvas = _slide_canvas(slide_id, coords_dir, slide_dir, downsample,
                           attn_vec.shape[0], patch_size)
    if canvas is None:
        return False
    coords, thumb, ds, stride = canvas

    subject = "this pathway" if pathway_idx is None else f"pathway {pathway_idx}"
    title = (f"{slide_id} — where {subject} looked"
             if pathway_idx is not None else f"{slide_id} — pathway attention")
    return _paint_and_save(attn_vec, coords, thumb, ds, stride, out_path, title,
                           subject, style, ATTENTION_SEMANTICS,
                           label=f"pathway {pathway_idx}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ckpt = _env("INTERP_CKPT")
    if not ckpt:
        raise SystemExit("Set INTERP_CKPT to a trained s_<fold>_checkpoint.pt")
    fold = _env("INTERP_FOLD", 0, int)
    case_idx = _env("INTERP_CASE_IDX", 0, int)
    case_id_wanted = _env("INTERP_CASE_ID")
    outdir = _env("INTERP_OUTDIR", "./interpret_out")
    topk = _env("INTERP_TOPK", 10, int)
    ig_steps = _env("INTERP_IG_STEPS", 50, int)
    coords_dir = _env("INTERP_COORDS_DIR")
    slide_dir = _env("INTERP_SLIDE_DIR")
    patch_size = _env("INTERP_PATCH_SIZE", 256, int)
    ig_reduce = _env("INTERP_IG_REDUCE", "signed_sum")
    style = _heatmap_style()

    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = _build_args()
    _, val_split = args.dataset_factory.return_splits(
        args, csv_path=f"{args.split_dir}/splits_{fold}.csv", fold=fold)

    # INTERP_CASE_ID wins over INTERP_CASE_IDX: the index is only stable for a
    # fixed feature directory (see _resolve_case_idx), the name always is.
    if case_id_wanted:
        if os.environ.get("INTERP_CASE_IDX"):
            print(f"[note] both INTERP_CASE_ID and INTERP_CASE_IDX are set; "
                  f"using INTERP_CASE_ID={case_id_wanted} and ignoring "
                  f"INTERP_CASE_IDX={case_idx}.")
        case_idx = _resolve_case_idx(val_split, case_id_wanted)

    model = _load_model(ckpt, args, val_split.omic_names, device)
    wsi, omics, slide_ids, case_id, event_time, censorship = _fetch_case(
        val_split, case_idx, device)
    print(f"Case {case_id} | slides={slide_ids} | patches={wsi.shape[1]} "
          f"| pathways={len(omics)}")

    # Pathway tokens are the signature CSV's columns in order, so the figures
    # can be titled with real names instead of an index. Same file the dataset
    # factory read, addressed the same way.
    pathway_names = PathwayNames.load(
        type_of_path=args.type_of_path).check(len(omics))

    # verify the spatial-overlay inputs (.h5 coords + slide) exist for this case
    files_ok = check_case_files(case_id, slide_ids, coords_dir, slide_dir)

    # (1) attention -------------------------------------------------------- #
    cross_attn_pathways, logits = extract_attention(model, wsi, omics)
    np.save(os.path.join(outdir, f"{case_id}_cross_attn_pathways.npy"),
            cross_attn_pathways)

    # Case-level facts the renderers caption with, written once next to the
    # arrays so they do not have to re-open the dataset to find them.
    risk = _risk_from_logits(logits)
    summary = {
        "case_id": case_id,
        "slide_ids": [str(s) for s in slide_ids],
        "risk": risk,
        "event_time": event_time,
        "censored": bool(censorship),
        "n_patches": int(wsi.shape[1]),
        "n_pathways": int(len(omics)),
        "checkpoint": ckpt,
        "fold": fold,
        "type_of_path": args.type_of_path,
    }
    print(f"Predicted risk {risk:+.2f} | "
          f"{'last follow-up' if censorship else 'survival time'} "
          f"{event_time:.0f} months")

    # (2) captum ----------------------------------------------------------- #
    try:
        import captum.attr  # noqa: F401  -- probe: is captum actually installed?
    except ImportError as e:
        import sys
        print(f"[warn] could not import captum.attr from this interpreter.\n"
              f"       python  : {sys.executable}\n"
              f"       reason  : {e!r}\n"
              f"       -> skipping importance ranking; ranking top pathways by "
              f"mean attention instead.")
        top_pathways = np.argsort(cross_attn_pathways.mean(axis=1))[::-1][:topk].tolist()
    else:
        path_df, gene_df, wsi_attr, delta = captum_importance(
            model, wsi, omics, val_split.omic_names, ig_steps, device,
            pathway_names=pathway_names,
            internal_batch_size=_env("INTERP_IG_BATCH", 8, int))
        path_df.to_csv(os.path.join(outdir, f"{case_id}_pathway_importance.csv"),
                       index=False)
        gene_df.to_csv(os.path.join(outdir, f"{case_id}_gene_importance.csv"),
                       index=False)
        # The CSV is ordered by signed effect (risk increases first) so it reads
        # as "what moved this case's risk, and which way". The heatmaps still
        # want the |IG| ranking, so take it explicitly rather than off row order.
        top_pathways = path_df.nlargest(topk, "abs_risk_delta")["pathway_idx"].tolist()
        print(f"Top pathways by |IG|: "
              + ", ".join(f"{p} ({pathway_names[p]})" for p in top_pathways))
        # |IG| says how much a pathway moved; it does not say which way. The
        # heatmaps below still key off the |IG| ranking (a pathway is worth
        # looking at spatially either way), but the direction is what the
        # write-up needs, so it is printed and columned, not left in the CSV.
        print_direction_report(path_df, topk)

        # (2b) IG as a spatial map ---------------------------------------- #
        ig_patch, ig_signed = patch_attribution(wsi_attr, ig_reduce)
        np.save(os.path.join(outdir, f"{case_id}_ig_patch_attr.npy"), ig_patch)

        wsi_total = float(ig_patch.sum())
        omics_total = float(path_df["risk_delta"].sum())
        summary.update({
            "ig_steps": ig_steps,
            "ig_reduce": ig_reduce,
            "ig_baseline": "zeros",
            "ig_convergence_delta": delta,
            "ig_wsi_total": wsi_total,
            "ig_omics_total": omics_total,
        })
        # IG's completeness axiom: every attribution together should account for
        # risk(case) - risk(baseline). Printing the split is how you tell a real
        # histology signal from a map of rounding error, and printing delta is
        # how you tell whether n_steps was enough to trust either number.
        print(f"IG (zeros baseline, {ig_steps} steps): WSI {wsi_total:+.3g} + "
              f"omics {omics_total:+.3g} = {wsi_total + omics_total:+.3g} "
              f"| convergence delta {delta:+.3g}")
        if abs(delta) > 0.05 * max(abs(wsi_total + omics_total), 1e-12):
            print(f"[warn] convergence delta is >5% of the attributed total — "
                  f"raise INTERP_IG_STEPS before reading the map quantitatively.")

        if files_ok and len(slide_ids) == 1:
            canvas = _slide_canvas(slide_ids[0], coords_dir, slide_dir,
                                   style.get("downsample", 32),
                                   ig_patch.shape[0], patch_size)
            if canvas is not None:
                coords, thumb, ds, stride = canvas
                istyle = ig_style(style, ig_signed)
                istyle.pop("downsample", None)
                pd.DataFrame({"patch_idx": np.arange(ig_patch.shape[0]),
                              "x": coords[:, 0], "y": coords[:, 1],
                              "ig_attr": ig_patch}).sort_values(
                    "ig_attr", key=np.abs, ascending=False).to_csv(
                        os.path.join(outdir, f"{case_id}_ig_patch_attr.csv"),
                        index=False)
                _paint_and_save(
                    ig_patch, coords, thumb, ds, stride,
                    os.path.join(outdir, f"{case_id}_ig_heatmap.png"),
                    title=f"{slide_ids[0]} — tissue driving the risk prediction "
                          f"(integrated gradients)",
                    subject="this case's risk score",
                    style=istyle, semantics=IG_SEMANTICS,
                    label="IG / WSI")

    # (3) overlay per top pathway ----------------------------------------- #
    if files_ok and len(slide_ids) == 1:
        for p in top_pathways:
            out_png = os.path.join(outdir, f"{case_id}_pathway{p}_heatmap.png")
            render_overlay(cross_attn_pathways[p], slide_ids[0],
                           coords_dir, slide_dir, patch_size, out_png,
                           pathway_idx=p, style=style)
    else:
        for p in top_pathways:
            np.save(os.path.join(outdir, f"{case_id}_pathway{p}_patch_attn.npy"),
                    cross_attn_pathways[p])
        print(
            "\n[info] Spatial overlays skipped. Saved per-patch attention "
            "vectors instead.\n"
            "       To paint them on the slide you need, per slide_id:\n"
            "         - a CLAM patch file '<slide_id>.h5' with a 'coords' "
            "dataset, and\n"
            "         - the whole-slide image '<slide_id>.svs'.\n"
            "       Regenerate coords with CLAM's create_patches_fp.py "
            "(same patch_size / step used for feature extraction), then set\n"
            "       INTERP_COORDS_DIR and INTERP_SLIDE_DIR and re-run.\n"
            "       (Multi-slide cases are also skipped; pick a single-slide "
            "case.)"
        )

    # Written last so it carries the IG totals the run actually produced, not
    # just the case-level facts known before Captum ran.
    with open(os.path.join(outdir, f"{case_id}_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Outputs in: {outdir}")


if __name__ == "__main__":
    main()
