"""Map where the top pathways (and their top genes) influenced the WSI.

A contact sheet: one panel per top pathway, all on one page, sharing a scale.
For the paper's per-pathway column layout (named header, mosaic, patch strip,
gene bars) use ``render_cross_modal.py`` instead — this one is for scanning many
pathways at once.

Standalone renderer that works purely from the files interpret_survpath.py
already wrote — no model, no dataset, no GPU. It needs:

    <case>_cross_attn_pathways.npy   [P, N]  pathway -> patch attention  (REQUIRED)
    <case>_pathway_importance.csv            to rank the top-K pathways   (optional*)
    <case>_gene_importance.csv               to label each pathway's genes(optional)

    a CLAM coords file  <slide_id>.h5  with a 'coords' dataset  (REQUIRED)
    the whole-slide image <slide_id>.svs                        (REQUIRED)
    the study metadata CSV, to map <case> -> <slide_id>          (REQUIRED**)

  * If pathway_importance.csv is missing, pathways are ranked by mean attention
    straight from the .npy. If gene_importance.csv is missing, panels are drawn
    without gene labels.
 ** Only to turn --case-id into the file name; --coords-h5/--slide-path or
    --slide-id make it unnecessary.

IMPORTANT — what this can and cannot show
-----------------------------------------
SurvPath produces a spatial (per-patch) attention map for each *pathway*, but
only a single scalar importance per *gene* (no spatial dimension). There is no
gene-level "where on the slide". So every gene inside a pathway necessarily
shares that pathway's patch footprint. This script therefore draws one heatmap
per top pathway and annotates it with that pathway's top genes — the honest
reading of "where the top genes influenced the WSI".

And a second caveat the figure now states for itself: rows of A_{P->H} are
highly correlated, so these panels tend to look alike. The caption reports the
measured cross-pathway rank correlation, plus a Moran's I test per panel of
whether the map is spatially structured at all — because rank normalisation
renders noise just as vividly as signal.

Output: a single multi-panel PNG (one panel per top pathway, plus an optional
gene-importance-weighted combined panel). Use --separate to also emit one PNG
per pathway.

Pointing it at the .h5 and .svs
-------------------------------
Simplest: name the two files outright. Nothing is derived, so no directories and
no metadata CSV are needed, and the exact on-disk name — UUID and all — is used
verbatim:

    python -m interpretation.render_gene_pathway_map \
        --interp-dir  results_brca/interpret \
        --case-id     TCGA-AC-A23E \
        --coords-h5   /data/patches_h5/TCGA-AC-A23E-01Z-00-DX1.A23982C3-E0EB-4DB2-84EE-26E0005E3F66.h5 \
        --slide-path  /data/slides/TCGA-AC-A23E-01Z-00-DX1.A23982C3-E0EB-4DB2-84EE-26E0005E3F66.svs

(--case-id is still required: it is the prefix of the .npy/.csv this reads out of
--interp-dir, not a file name for the slide.)

Otherwise it looks the files up by name, exactly as interpret_survpath.py does:
"<slide_id>.h5" under --coords-dir and "<slide_id>.svs" under --slide-dir. The
stem is the case's slide_id from the study metadata CSV, via the same
case_id -> slide_id chain the dataset uses — a case id (TCGA-AC-A23E) is NOT a
file name, the files are named by slide id
(TCGA-AC-A23E-01Z-00-DX1.<uuid>), which is why looking up the case id found
nothing. --slide-id supplies that stem directly and skips the CSV; give it in
full, UUID included.

    INTERP_COORDS_DIR  dir of CLAM patch files "<slide_id>.h5"
    INTERP_SLIDE_DIR   dir of the whole-slide images "<slide_id>.svs"
    INTERP_LABEL_FILE  study metadata CSV mapping case_id -> slide_id
    INTERP_PATCH_SIZE  patch edge in level-0 pixels        (default: inferred)
    INTERP_OUTDIR      where interpret_survpath.py wrote its .npy / .csv

The matching CLI flags (--coords-dir / --slide-dir / --label-file /
--patch-size / --interp-dir) override the environment when given.

    export INTERP_COORDS_DIR=/path/to/CLAM/patches_h5
    export INTERP_SLIDE_DIR=/path/to/slides
    python -m interpretation.render_gene_pathway_map \
        --interp-dir  results_brca/interpret \
        --case-id     TCGA-AC-A23E \
        --label-file  datasets_csv/metadata/tcga_brca.csv
"""
import argparse
import math
import os

import numpy as np

from interpretation import case_inputs as ci
from interpretation import heatmap_utils as hm
from interpretation.pathway_names import PathwayNames


# --------------------------------------------------------------------------- #
# painting
# --------------------------------------------------------------------------- #
def _panel_title(rank, pathway_idx, name, genes, scale=None, diag=None):
    head = f"#{rank}  {name}" if name else f"#{rank}  pathway {pathway_idx}"
    if scale is not None:
        head += f"\nraw {scale.raw_min:+.2f} to {scale.raw_max:+.2f}"
    if diag is not None:
        # Per-panel, because whether a map is real varies panel to panel and the
        # shared caption can only speak about the set.
        head += (f"   ·   I={diag.morans_i:+.2f}"
                 f"{'' if diag.structured else ' (noise)'}")
    if not genes:
        return head
    return f"{head}\ntop genes: {', '.join(g for g, _ in genes)}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ci.add_input_args(ap)
    ap.add_argument("--out", default=None, help="output PNG (default in --interp-dir)")
    ap.add_argument("--top-pathways", type=int, default=5)
    ap.add_argument("--top-genes", type=int, default=5)
    ap.add_argument("--type-of-path", default=ci.env("INTERP_TYPE_OF_PATH", "combine"),
                    help="which <type>_signatures.csv names the pathways "
                         "(combine / hallmarks / xena) [env: INTERP_TYPE_OF_PATH]")
    ap.add_argument("--metadata-dir", default=ci.env("INTERP_METADATA_DIR"),
                    help="dir holding <type_of_path>_signatures.csv "
                         "(default datasets_csv/metadata)")
    ap.add_argument("--cmap", default=hm.DEFAULT_CMAP,
                    help=f"'{hm.DEFAULT_CMAP}' (sequential blue, default), "
                         f"'{hm.DEFAULT_SIGNED_CMAP}' (diverging; use with "
                         f"--norm signed), 'jet' to match the published figure, "
                         f"or any matplotlib name")
    ap.add_argument("--render", default="overlay", choices=("overlay", "mosaic"),
                    help="overlay (default) blends onto the H&E; mosaic draws "
                         "opaque tiles on a flat canvas, as published")
    ap.add_argument("--texture", type=float, default=0.0,
                    help="0..1 — shade mosaic tiles by H&E luminance")
    ap.add_argument("--norm", default="rank",
                    choices=("rank", "percentile", "minmax", "signed"),
                    help="rank (default) spreads the bell-shaped pre-softmax "
                         "scores over the whole ramp so structure is visible; "
                         "percentile keeps raw magnitude with outliers clipped")
    ap.add_argument("--pct-lo", type=float, default=1.0,
                    help="low clip percentile for --norm percentile")
    ap.add_argument("--pct-hi", type=float, default=99.0,
                    help="high clip percentile for --norm percentile/signed")
    ap.add_argument("--smooth", type=float, default=0.5,
                    help="Gaussian sigma in tile widths; 0 = crisp tiles")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="opacity of the hottest tissue (coldest fades out)")
    ap.add_argument("--alpha-min", type=float, default=0.0,
                    help="opacity of the coldest tissue")
    ap.add_argument("--alpha-gamma", type=float, default=1.5,
                    help=">1 concentrates ink on the high-attention tail")
    ap.add_argument("--focus-top", type=float, default=None,
                    help="paint only the top N%% of the range, leaving the rest "
                         "as plain H&E (e.g. 20)")
    ap.add_argument("--separate", action="store_true",
                    help="also write one PNG per pathway")
    ap.add_argument("--no-combined", action="store_true",
                    help="skip the gene-importance-weighted combined panel")
    args = ap.parse_args()

    # --coords-dir / --slide-dir are only needed for the by-name lookup, so
    # ci.resolve_files checks those; --interp-dir is where the .npy/.csv live
    # and is needed either way.
    if not args.interp_dir:
        raise SystemExit("missing required path: --interp-dir (or INTERP_OUTDIR)")
    if args.label_file and not os.path.isfile(args.label_file):
        raise SystemExit(f"--label-file: no such file: {args.label_file}")

    if args.norm == "signed" and args.cmap == hm.DEFAULT_CMAP:
        args.cmap = hm.DEFAULT_SIGNED_CMAP     # signed scores need two poles

    style = dict(cmap_name=args.cmap, norm_mode=args.norm, pct_lo=args.pct_lo,
                 pct_hi=args.pct_hi, smooth_tiles=args.smooth,
                 alpha_max=args.alpha, alpha_min=args.alpha_min,
                 alpha_gamma=args.alpha_gamma, focus_top=args.focus_top,
                 render=args.render, texture=args.texture)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5_path, slide_path, slide_id = ci.resolve_files(args)
    cross_attn = ci.load_cross_attn(args.interp_dir, args.case_id, args.cross_attn)

    slide, thumb, coords, ds = ci.open_slide_and_coords(
        h5_path, slide_path, args.downsample)
    slide.close()                      # only the thumbnail is needed here
    print(f"[info] slide '{slide_id}': thumbnail {thumb.shape[1]}x{thumb.shape[0]} "
          f"px, downsample {ds:.1f}, {coords.shape[0]} patches")

    # Attention is indexed by patch, so a coords file from a different
    # slide/patching run cannot be painted with it. Catch it rather than
    # mis-plotting.
    ci.check_patch_alignment(coords, cross_attn, h5_path)

    # Tiles must cover the grid the features came from, not an assumed 256 px —
    # a too-small footprint is what reduced these maps to isolated dots.
    stride = hm.infer_patch_stride(coords, declared=args.patch_size)

    path_df = ci.load_pathway_importance(args.interp_dir, args.case_id)
    gene_df = ci.load_gene_importance(args.interp_dir, args.case_id)
    tops = ci.top_pathways(path_df, cross_attn, args.top_pathways)
    print(f"[info] top {len(tops)} pathways: {tops}")

    names = PathwayNames.from_importance_csv(path_df)
    if not names:
        names = PathwayNames.load(type_of_path=args.type_of_path,
                                  metadata_dir=args.metadata_dir)
    names = names.check(cross_attn.shape[0])

    spec = hm.specificity_report(cross_attn, tops)
    print(f"[info] {spec.summary()}")

    # per-pathway panels ---------------------------------------------------- #
    panels = []          # (title, overlay_image)
    last_scale = None
    weighted = np.zeros(cross_attn.shape[1], dtype=np.float64)
    for rank, p in enumerate(tops, start=1):
        genes = ci.top_genes(gene_df, p, args.top_genes)
        attn = cross_attn[p]
        overlay, scale = hm.paint_attention(thumb, coords, ds, attn, stride, **style)
        diag = hm.spatial_diagnostics(coords, attn, stride)
        print(f"[info] pathway {p} ({names[p]}): {diag.strength}, "
              f"I={diag.morans_i:+.3f} z={diag.z:+.1f}")
        last_scale = scale
        panels.append((_panel_title(rank, p, names[p] if names else None,
                                    genes, scale, diag), overlay))

        # combined map: weight this pathway's (normalised) attention by the
        # summed |IG| of its top genes, so pathways with stronger gene drivers
        # contribute more to the blended picture.
        w = sum(abs(a) for _, a in genes) if genes else float(np.abs(attn).sum())
        a = attn.astype(np.float64)
        a = (a - a.min()) / (np.ptp(a) + 1e-8)
        weighted += w * a

        if args.separate:
            sep = os.path.join(args.interp_dir,
                               f"{args.case_id}_pathway{p}_genes_heatmap.png")
            subject = names[p] if names else f"pathway {p}"
            if genes:
                subject += f" (top genes {', '.join(g for g, _ in genes)})"
            hm.save_overlay_figure(
                overlay, scale, sep,
                title=f"{args.case_id} — #{rank}: where {subject} looked",
                caption=hm.color_caption(scale, cmap_name=args.cmap,
                                         stride_l0=stride, ds=ds,
                                         subject=names[p] if names
                                                 else f"pathway {p}",
                                         focus_top=args.focus_top,
                                         diagnostics=diag),
                cmap_name=args.cmap,
                subtitle=f"{coords.shape[0]} tiles · {stride} px at level 0 · "
                         f"thumbnail downsample {ds:.0f}x")
            print(f"[ok] wrote {sep}")

    if not args.no_combined:
        combined_overlay, combined_scale = hm.paint_attention(
            thumb, coords, ds, weighted, stride, **style)
        panels.append(("combined (gene-|IG|-weighted across top pathways)",
                       combined_overlay))
        last_scale = last_scale or combined_scale

    # multi-panel figure ---------------------------------------------------- #
    # One shared colorbar + caption rather than per-panel legends: every panel
    # is normalised the same way, so the ramp means the same thing in each. The
    # per-panel raw range and Moran's I are what differ, and they are in the
    # panel titles.
    n = len(panels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    panel_w = 6.0
    panel_h = panel_w * (thumb.shape[0] / max(thumb.shape[1], 1)) + 1.1
    caption = hm.color_caption(
        last_scale, cmap_name=args.cmap, stride_l0=stride, ds=ds,
        subject="each pathway", focus_top=args.focus_top, width=150,
        specificity=spec)
    cap_h = 0.19 * (caption.count("\n") + 1) + 0.3
    grid_h = panel_h * nrows
    fig_h = grid_h + cap_h + 1.5

    fig = plt.figure(figsize=(panel_w * ncols, fig_h), facecolor=hm.SURFACE)
    grid_bottom = (cap_h + 0.85) / fig_h
    gs = fig.add_gridspec(nrows, ncols, left=0.01, right=0.99,
                          bottom=grid_bottom, top=1.0 - 0.75 / fig_h,
                          wspace=0.03, hspace=0.16)
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, color=hm.INK_SECONDARY)
        ax.set_axis_off()

    fig.suptitle(f"{args.case_id} — where the top {len(tops)} pathways looked "
                 f"(annotated with top {args.top_genes} genes each)",
                 fontsize=13, color=hm.INK, y=1.0 - 0.22 / fig_h)
    hm.add_colorbar(fig, [0.35, (cap_h + 0.45) / fig_h, 0.30, 0.16 / fig_h],
                    last_scale, args.cmap)
    fig.text(0.5, (cap_h - 0.14) / fig_h, caption, ha="center", va="top",
             color=hm.INK_SECONDARY, fontsize=8.5, linespacing=1.5)

    out = args.out or os.path.join(
        args.interp_dir, f"{args.case_id}_top_pathways_genes_map.png")
    fig.savefig(out, dpi=200, facecolor=hm.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
