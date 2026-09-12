"""Locating and loading everything a case's interpretability figure needs.

``render_gene_pathway_map.py`` and ``render_cross_modal.py`` draw very different
figures from an identical set of inputs — the .npy/.csv that
``interpret_survpath.py`` wrote, plus a CLAM coords .h5 and the slide itself.
Resolving those was duplicated between them, and the case_id -> slide_id chain
is fiddly enough (see ``resolve_slide_id``) that two copies would inevitably
drift. It lives here once; both renderers share the CLI flags too, via
``add_input_args``.

Nothing in this module draws anything.
"""
from __future__ import annotations

import json
import os

import numpy as np
if not hasattr(np, "typeDict"):      # removed in NumPy 1.24; older h5py uses it
    np.typeDict = np.sctypeDict
import pandas as pd

from interpretation import heatmap_utils as hm

SLIDE_EXTS = hm.SLIDE_EXTS


def env(name, default=None, cast=str):
    """Env lookup where an empty string counts as unset (as interpret_survpath)."""
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def add_input_args(ap):
    """Add the flags that say *which case* and *which files* to both renderers."""
    ap.add_argument("--interp-dir", default=env("INTERP_OUTDIR"),
                    help="dir holding the .npy / .csv from interpret_survpath.py "
                         "[env: INTERP_OUTDIR]")
    ap.add_argument("--case-id", required=True,
                    help="e.g. TCGA-AC-A23E — also the prefix of the .npy/.csv "
                         "interpret_survpath.py wrote in --interp-dir")
    ap.add_argument("--label-file", default=env("INTERP_LABEL_FILE"),
                    help="study metadata CSV (the same --label_file main.py and "
                         "interpret_survpath.py get, e.g. "
                         "datasets_csv/metadata/tcga_brca.csv). Used to map "
                         "--case-id to the slide id the .h5/.svs are named after "
                         "[env: INTERP_LABEL_FILE]")
    ap.add_argument("--slide-id", default=None,
                    help="the name the .h5/.svs actually carry, minus the "
                         "extension — in full, UUID included. Overrides the "
                         "--label-file lookup; required for multi-slide cases.")
    ap.add_argument("--coords-dir", default=env("INTERP_COORDS_DIR"),
                    help="dir of CLAM '<slide_id>.h5' [env: INTERP_COORDS_DIR]")
    ap.add_argument("--slide-dir", default=env("INTERP_SLIDE_DIR"),
                    help="dir of whole-slide images [env: INTERP_SLIDE_DIR]")
    ap.add_argument("--coords-h5", default=None,
                    help="path to the CLAM coords .h5 itself. Skips the "
                         "case_id -> slide_id -> file-name chain entirely; give "
                         "it with --slide-path and neither --coords-dir/"
                         "--slide-dir nor --label-file is consulted.")
    ap.add_argument("--slide-path", default=None,
                    help="path to the whole-slide image itself (see --coords-h5)")
    ap.add_argument("--cross-attn", default=None,
                    help="override path to <case>_cross_attn_pathways.npy")
    ap.add_argument("--patch-size", type=int,
                    default=env("INTERP_PATCH_SIZE", None, int),
                    help="patch edge in level-0 px [env: INTERP_PATCH_SIZE]. "
                         "Normally leave unset: the stride is inferred from the "
                         "coords grid so tiles tile the slide without gaps.")
    ap.add_argument("--downsample", type=float, default=32,
                    help="thumbnail downsample; lower = more detail, bigger PNG")
    return ap


# --------------------------------------------------------------------------- #
# case id -> slide id -> files
# --------------------------------------------------------------------------- #
def resolve_slide_id(label_file, case_id, explicit=None):
    """Get the exact slide id whose .h5/.svs we should open.

    interpret_survpath.py never guesses this stem: the dataset reads it out of
    the study metadata CSV and it goes straight to the file lookup. So do we —
    ``hm.slide_ids_for_case`` reproduces that CSV -> slide_id chain, turning a
    case id (TCGA-AC-A23E) into the name the files actually carry
    (TCGA-AC-A23E-01Z-00-DX1.<uuid>). ``explicit`` (--slide-id) overrides it.
    """
    if explicit:
        return str(explicit)

    slide_ids = hm.slide_ids_for_case(label_file, case_id)
    if not slide_ids:
        if not label_file:
            raise SystemExit(
                "cannot turn a case id into a slide id without the metadata "
                "CSV.\n"
                "  Pass --label-file (or export INTERP_LABEL_FILE), e.g. "
                "datasets_csv/metadata/tcga_brca.csv — the same file "
                "interpret_survpath.py gets as --label_file.\n"
                "  Or pass --slide-id directly (interpret_survpath.py prints it: "
                "'Case <case_id> | slides=[...]').")
        raise SystemExit(
            f"case '{case_id}' has no slide_id in {label_file}.\n"
            f"  Check the --case-id spelling against the 'case_id' column, or "
            f"pass --slide-id directly.")

    if len(slide_ids) > 1:
        # interpret_survpath.py skips multi-slide cases outright: one attention
        # vector spans the concatenated patches of every slide, so no single
        # coords file lines up with it.
        listing = "\n".join(f"    {s}" for s in slide_ids)
        raise SystemExit(
            f"case {case_id} has {len(slide_ids)} slides:\n{listing}\n"
            f"  A single coords file cannot cover them, so pass --slide-id with "
            f"the one you want (or pick a single-slide case).")

    print(f"[info] case {case_id} -> slide_id '{slide_ids[0]}' (from {label_file})")
    return slide_ids[0]


def resolve_files(args):
    """(h5_path, slide_path, slide_id) for the case we are about to paint.

    Two ways in, and the explicit one short-circuits everything:

      --coords-h5 / --slide-path   the files themselves. Nothing is derived, so
                                   no --coords-dir, --slide-dir, --label-file or
                                   --case-id -> slide_id chain is consulted.
      --slide-id / --case-id       look up '<stem>.h5' / '<stem>.svs' under
                                   --coords-dir / --slide-dir.

    Pass names exactly as they are on disk, UUID and all — nothing here rewrites
    or shortens them.
    """
    for flag, path in (("--coords-h5", args.coords_h5),
                       ("--slide-path", args.slide_path)):
        if path and not os.path.isfile(path):
            raise SystemExit(f"{flag}: no such file: {path}")

    if args.coords_h5 and args.slide_path:
        # slide_id is only a label for the log/figure here — nothing is opened
        # by name, so a basename is enough.
        slide_id = args.slide_id or os.path.splitext(
            os.path.basename(args.coords_h5))[0]
        print(f"[info] using the files given: h5={args.coords_h5}  "
              f"slide={args.slide_path}")
        return args.coords_h5, args.slide_path, slide_id

    needed = [f"--{n} (or {e})" for n, e, val, want in (
        ("coords-dir", "INTERP_COORDS_DIR", args.coords_dir, not args.coords_h5),
        ("slide-dir", "INTERP_SLIDE_DIR", args.slide_dir, not args.slide_path),
    ) if want and not val]
    if needed:
        raise SystemExit(
            "missing required path(s): " + ", ".join(needed) + "\n"
            "  Or skip the lookup entirely and name the two files directly:\n"
            "    --coords-h5 /path/to/<slide_id>.h5 "
            "--slide-path /path/to/<slide_id>.svs")

    slide_id = resolve_slide_id(args.label_file, args.case_id, args.slide_id)
    h5_path = args.coords_h5 or hm.find_case_file(args.coords_dir, slide_id, (".h5",))
    slide_path = args.slide_path or hm.find_case_file(
        args.slide_dir, slide_id, SLIDE_EXTS)
    if h5_path is None or slide_path is None:
        hm.check_case_files(slide_id, [slide_id], args.coords_dir, args.slide_dir)
        raise SystemExit(
            f"coords/slide not found for '{slide_id}' (see the [check] lines "
            f"above): h5={h5_path}, slide={slide_path}\n"
            f"  If the files on disk carry a name this did not reproduce, pass "
            f"them outright:\n"
            f"    --coords-h5 <that exact .h5> --slide-path <that exact .svs>")
    return h5_path, slide_path, slide_id


# --------------------------------------------------------------------------- #
# slide + coords
# --------------------------------------------------------------------------- #
def open_slide_and_coords(h5_path, slide_path, downsample):
    """(slide, thumb, coords, ds) — the open handle is returned for patch reads.

    Callers that want representative patch crops need the OpenSlide object, not
    just the thumbnail, so it stays open and is theirs to close.
    """
    import h5py
    import openslide

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]                        # [N, 2] level-0 px

    slide = openslide.OpenSlide(slide_path)
    level = slide.get_best_level_for_downsample(downsample)
    ds = slide.level_downsamples[level]
    thumb = np.array(
        slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB"))
    return slide, thumb, coords, ds


def read_patch(slide, x, y, stride_l0, out_px=128):
    """One square patch at level-0 (x, y), returned as an ``out_px`` RGB array.

    Reads from the coarsest pyramid level that still has ``out_px`` of detail —
    reading a 1024 px region at level 0 to shrink it to 128 costs 64x the pixels
    for no visible gain, and a figure does a few dozen of these.
    """
    from PIL import Image

    level = slide.get_best_level_for_downsample(max(stride_l0 / out_px, 1.0))
    lds = slide.level_downsamples[level]
    size = max(int(round(stride_l0 / lds)), 1)
    region = slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
    if region.size != (out_px, out_px):
        region = region.resize((out_px, out_px), Image.LANCZOS)
    return np.asarray(region)


def pick_representative_patches(coords, scores, stride_l0, n=4, min_sep_tiles=3.0):
    """Indices of the top-scoring patches, spread out over the slide.

    Taking the literal top-n almost always returns n adjacent tiles from the
    single hottest blob, which shows the reader the same picture four times.
    Enforcing a minimum separation (in tile widths) turns the strip into four
    genuinely different views of what this pathway attends to. Falls back to the
    plain top-n if the constraint cannot be met.
    """
    coords = np.asarray(coords, dtype=np.float64)
    order = np.argsort(np.asarray(scores).ravel())[::-1]
    min_sep = float(min_sep_tiles) * float(stride_l0)

    chosen = []
    for i in order:
        if len(chosen) >= n:
            break
        if all(np.hypot(*(coords[i] - coords[j])) >= min_sep for j in chosen):
            chosen.append(int(i))

    for i in order:                       # top up if separation was too strict
        if len(chosen) >= n:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    return chosen[:n]


# --------------------------------------------------------------------------- #
# what interpret_survpath.py wrote
# --------------------------------------------------------------------------- #
def load_cross_attn(interp_dir, case_id, override=None):
    """The [P, N] pathway -> patch matrix."""
    path = override or os.path.join(interp_dir,
                                    f"{case_id}_cross_attn_pathways.npy")
    if not os.path.isfile(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  This script needs the full pathway->patch matrix. Re-run "
            f"interpret_survpath.py (it always saves "
            f"*_cross_attn_pathways.npy), or pass --cross-attn.")
    a = np.load(path)
    if a.ndim != 2:
        raise SystemExit(f"expected [P, N] matrix, got shape {a.shape}")
    print(f"[info] cross_attn {a.shape} (pathways x patches)")
    return a


def load_pathway_importance(interp_dir, case_id):
    """The Captum pathway ranking, or None if IG was skipped."""
    csv = os.path.join(interp_dir, f"{case_id}_pathway_importance.csv")
    if not os.path.isfile(csv):
        return None
    return pd.read_csv(csv)


def load_gene_importance(interp_dir, case_id):
    """The Captum gene ranking, or None if IG was skipped."""
    csv = os.path.join(interp_dir, f"{case_id}_gene_importance.csv")
    if not os.path.isfile(csv):
        return None
    return pd.read_csv(csv)


def load_summary(interp_dir, case_id):
    """Risk / survival metadata written alongside the arrays, or {}."""
    path = os.path.join(interp_dir, f"{case_id}_summary.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_pathways(path_df, cross_attn, k):
    """Pathway indices, most influential first (|IG|, else mean attention).

    ``abs_attr_sum`` is what runs before the risk-units rename wrote; both names
    hold the same number, so old interpret dirs still rank rather than silently
    falling back to attention.
    """
    col = next((c for c in ("abs_risk_delta", "abs_attr_sum")
                if path_df is not None and c in path_df), None)
    if col is not None:
        df = path_df.sort_values(col, ascending=False)
        return df["pathway_idx"].head(k).astype(int).tolist()
    print("[warn] no pathway_importance.csv -> ranking pathways by mean attention.")
    return np.argsort(cross_attn.mean(axis=1))[::-1][:k].astype(int).tolist()


def top_genes(gene_df, pathway_idx, k):
    """[(gene, signed_attr), ...] for one pathway, largest |attr| first.

    Signed, because the bar chart's whole point is direction: a gene that pushes
    predicted risk down is as interesting as one that pushes it up.
    """
    if gene_df is None or "pathway_idx" not in gene_df:
        return []
    df = gene_df[gene_df["pathway_idx"] == pathway_idx]
    if df.empty:
        return []
    col = "attr" if "attr" in df else "abs_attr"
    df = df.reindex(df[col].abs().sort_values(ascending=False).index)
    return [(str(g), float(v))
            for g, v in df[["gene", col]].head(k).itertuples(index=False, name=None)]


def check_patch_alignment(coords, cross_attn, h5_path):
    """Attention is indexed by patch, so a mismatched coords file cannot paint it."""
    if coords.shape[0] != cross_attn.shape[1]:
        raise SystemExit(
            f"coords ({coords.shape[0]} patches) != attention "
            f"({cross_attn.shape[1]} patches).\n"
            f"  {h5_path} does not match the features these attentions came "
            f"from — wrong slide, a different patch_size/step, or a multi-slide "
            f"case. Pass --coords-h5 with the right slide's coords file.")
