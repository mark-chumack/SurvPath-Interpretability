"""The published "Interpret Cross-modal (A_{P->H})" block, one column per pathway.

Reproduces the layout of the SurvPath paper's cross-modal panel. Each column is
four stacked pieces for one pathway:

    Epithelial Mesenchymal Transition   <- the signature CSV's column name
    ------------------------------
    [ tissue-shaped mosaic on white ]   <- that pathway's row of A_{P->H}
    [patch][patch][patch][patch]        <- H&E at its top-attended coords
    ANPEP  ####|                        <- top genes by signed Captum |IG|
    IL6        |###                        red = raises risk, blue = lowers it
      -0.004    0    0.006

Run it on what ``interpret_survpath.py`` already wrote — no model, no GPU:

    python -m interpretation.render_cross_modal \
        --interp-dir results_brca/interpret \
        --case-id    TCGA-BH-A0DI \
        --coords-h5  /data/patches_h5/TCGA-BH-A0DI-01Z-00-DX1.<uuid>.h5 \
        --slide-path /data/slides/TCGA-BH-A0DI-01Z-00-DX1.<uuid>.svs

(or set INTERP_COORDS_DIR / INTERP_SLIDE_DIR / INTERP_LABEL_FILE and let it look
the files up by name — see ``interpretation.case_inputs``.)

Two deliberate departures from this repo's usual chart rules
------------------------------------------------------------
JET.  ``heatmap_utils`` rejects rainbow ramps, and for good reason: jet's
lightness is not monotone, so a reader cannot rank two mid-range tiles by eye.
This module defaults to it anyway because the point of the figure is to match
the published one. The cost is paid down in the caption, which states the ramp
order explicitly and tells the reader to use the colorbar for magnitude rather
than the hue. ``--cmap attention`` switches to the sequential blue ramp and
gives up the resemblance.

RAW ROWS.  Rows of A_{P->H} typically rank-correlate above 0.9, because the
dominant term in q_p . k_n is how attendable patch n is at all rather than which
pathway is asking. So these panels come out looking alike. That is a true
property of the attention, not a plotting artifact, and the fix is not to
subtract it away behind the reader's back — it is to measure it. Every figure
carries the measured cross-pathway correlation and a Moran's I test of whether
each map is spatially structured at all, so "these five look identical" and
"the median is -0.07" are answered on the figure instead of being left for the
reader to suspect.
"""
from __future__ import annotations

import argparse
import os
import textwrap

import numpy as np

from interpretation import case_inputs as ci
from interpretation import heatmap_utils as hm
from interpretation.pathway_names import PathwayNames

# Warm panel behind the whole block, as in the paper. Column tags are the
# documented categorical hues in their fixed order; they identify a column, they
# do not encode anything, so the pathway name itself stays in primary ink and
# only the rule under it is colored.
PANEL_BG = "#fbf3e4"
PANEL_EDGE = "#d9cfb8"
COLUMN_TAGS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")

DEFAULT_CMAP = "jet"


# --------------------------------------------------------------------------- #
# small drawing helpers
# --------------------------------------------------------------------------- #
def _patch_strip(slide, coords, idxs, stride_l0, out_px=128, gap=6):
    """The four chosen patches side by side as one image, with white gutters."""
    tiles = [ci.read_patch(slide, coords[i][0], coords[i][1], stride_l0, out_px)
             for i in idxs]
    if not tiles:
        return None
    strip = np.full((out_px, len(tiles) * out_px + (len(tiles) - 1) * gap, 3),
                    255, dtype=np.uint8)
    for k, t in enumerate(tiles):
        x = k * (out_px + gap)
        strip[:, x:x + out_px] = t
    return strip


def _tick_fmt(v):
    """Axis label with just enough precision for these tiny attributions."""
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 0.1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def _gene_bars(ax, genes):
    """Horizontal diverging bars: red raises predicted risk, blue lowers it.

    Signed Captum attributions, so direction is the message; magnitude is the
    length. Two hues with a zero line rather than one hue, because "which way"
    is not an ordinal quantity.
    """
    if not genes:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no gene attributions\n(captum not run)",
                ha="center", va="center", fontsize=6.5, color=hm.INK_MUTED,
                transform=ax.transAxes)
        return

    names = [g for g, _ in genes]
    vals = [v for _, v in genes]
    y = np.arange(len(vals))

    ax.barh(y, vals, height=0.62,
            color=[hm.POLE_POS if v >= 0 else hm.POLE_NEG for v in vals],
            linewidth=0)
    ax.axvline(0, color=hm.INK_MUTED, linewidth=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5, color=hm.INK)
    ax.invert_yaxis()                       # largest |attr| on top
    ax.set_ylim(len(vals) - 0.5, -0.5)

    span = max(abs(min(vals)), abs(max(vals))) or 1e-6
    lo = min(min(vals), 0.0) - 0.12 * span
    hi = max(max(vals), 0.0) + 0.12 * span
    ax.set_xlim(lo, hi)
    ax.set_xticks([lo + 0.06 * span, 0.0, hi - 0.06 * span])
    ax.set_xticklabels([_tick_fmt(lo + 0.06 * span), "0",
                        _tick_fmt(hi - 0.06 * span)], fontsize=6, color=hm.INK_MUTED)

    ax.tick_params(axis="both", length=0, pad=2)
    for side, sp in ax.spines.items():
        sp.set_color(hm.HAIRLINE if side in ("top", "right") else hm.INK_MUTED)
        sp.set_linewidth(0.6)
    ax.set_facecolor("#ffffff")
    ax.grid(False)


def _column_header(ax, name, tag):
    """Pathway name over a colored rule — the paper's per-column heading."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.42, name, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=hm.INK, linespacing=1.25)
    ax.plot([0.04, 0.96], [0.06, 0.06], color=tag, linewidth=2.2,
            solid_capstyle="round", clip_on=False)


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def render(case_id, cross_attn, coords, thumb, ds, slide, stride, tops,
           names, gene_df, out_path, cmap_name=DEFAULT_CMAP, norm_mode="rank",
           smooth=0.0, texture=0.0, n_genes=5, n_patches=4, patch_px=128,
           title=None, subtitle=None, dpi=220, col_width=3.3):
    """Draw the whole block and write it to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = len(tops)
    aspect = thumb.shape[0] / max(thumb.shape[1], 1)

    # --- paint each column's pieces before laying anything out ------------- #
    columns = []
    for rank, p in enumerate(tops):
        attn = cross_attn[p]
        img, scale = hm.paint_attention(
            thumb, coords, ds, attn, stride, cmap_name=cmap_name,
            norm_mode=norm_mode, smooth_tiles=smooth, render="mosaic",
            texture=texture, warn_cmap=False)      # jet here is deliberate
        diag = hm.spatial_diagnostics(coords, attn, stride)
        keep = ci.pick_representative_patches(coords, attn, stride, n=n_patches)
        strip = _patch_strip(slide, coords, keep, stride, out_px=patch_px)
        columns.append(dict(idx=p, rank=rank, img=img, scale=scale, diag=diag,
                            strip=strip, genes=ci.top_genes(gene_df, p, n_genes)))
        print(f"[info] pathway {p} ({names[p]}): {diag.strength}, "
              f"I={diag.morans_i:+.3f} z={diag.z:+.1f}")

    spec = hm.specificity_report(cross_attn, tops)
    print(f"[info] {spec.summary()}")

    # --- geometry ---------------------------------------------------------- #
    head_h = 0.62
    map_h = col_width * aspect
    strip_h = col_width / max(n_patches, 1) * 1.02
    bars_h = 0.30 * max(n_genes, 1) + 0.34
    title_h = 0.52 + (0.24 if subtitle else 0.0)

    caption = _caption(columns, spec, cmap_name, stride, ds,
                       width=int(26 * ncols) + 34)
    cap_h = 0.155 * (caption.count("\n") + 1) + 0.30
    bar_h = 0.62                                    # shared colorbar + label

    grid_h = head_h + map_h + strip_h + bars_h
    fig_w = col_width * ncols
    fig_h = title_h + grid_h + bar_h + cap_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PANEL_BG)
    fig.patch.set_edgecolor(PANEL_EDGE)
    fig.patch.set_linewidth(1.2)

    gs = fig.add_gridspec(
        4, ncols,
        height_ratios=[head_h, map_h, strip_h, bars_h],
        left=0.035, right=0.965,
        bottom=(cap_h + bar_h) / fig_h, top=1.0 - title_h / fig_h,
        wspace=0.16, hspace=0.06)

    for k, col in enumerate(columns):
        tag = COLUMN_TAGS[k % len(COLUMN_TAGS)]
        _column_header(fig.add_subplot(gs[0, k]),
                       names.wrapped(col["idx"], width=26), tag)

        ax_map = fig.add_subplot(gs[1, k])
        ax_map.imshow(col["img"], interpolation="nearest")
        ax_map.set_axis_off()
        if not col["diag"].structured:
            # Do not let a vivid panel imply a finding the test just rejected.
            ax_map.text(0.5, -0.02, "no spatial structure (see caption)",
                        transform=ax_map.transAxes, ha="center", va="top",
                        fontsize=6.5, color=hm.POLE_POS)

        ax_strip = fig.add_subplot(gs[2, k])
        ax_strip.set_axis_off()
        if col["strip"] is not None:
            ax_strip.imshow(col["strip"], interpolation="nearest")

        _gene_bars(fig.add_subplot(gs[3, k]), col["genes"])

    fig.suptitle(
        title or r"Interpret Cross-modal ($\mathrm{A}_{\mathcal{P}\rightarrow\mathcal{H}}$)",
        fontsize=13, color=hm.INK, fontweight="bold", y=1.0 - 0.30 / fig_h)
    if subtitle:
        fig.text(0.5, 1.0 - 0.54 / fig_h, subtitle, ha="center", va="top",
                 fontsize=8.5, color=hm.INK_SECONDARY)

    hm.add_colorbar(fig, [0.34, (cap_h + 0.30) / fig_h, 0.32, 0.13 / fig_h],
                    columns[0]["scale"], cmap_name)
    fig.text(0.5, (cap_h - 0.12) / fig_h, caption, ha="center", va="top",
             color=hm.INK_SECONDARY, fontsize=7.4, linespacing=1.45)

    fig.savefig(out_path, dpi=dpi, facecolor=PANEL_BG, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _caption(columns, spec, cmap_name, stride, ds, width=120):
    """The honest footnote: what the colors mean, and whether to believe them."""
    scale = columns[0]["scale"]
    lines = [
        hm.color_caption(scale, cmap_name=cmap_name, stride_l0=stride, ds=ds,
                         subject="the pathway named above", width=width),
        "Patches below each map are its top-attended tiles, spread apart so the "
        "four are not one blob seen four times. Bars are that pathway's genes by "
        "signed Captum |IG| on predicted risk: "
        f"red ({hm.POLE_POS}) raises risk, blue ({hm.POLE_NEG}) lowers it.",
        spec.summary(),
    ]

    weak = [c for c in columns if not c["diag"].structured]
    if weak:
        which = ", ".join(str(c["idx"]) for c in weak)
        lines.append(
            f"SIGNAL: pathway {which} failed the spatial-structure test "
            f"(Moran's I within 2 sigma of a shuffled-patch null), so those "
            f"panels are the ramp spreading noise over the section — the vivid "
            f"color is the normalisation, not a finding.")
    strong = [c for c in columns if c["diag"].structured]
    if strong:
        detail = "; ".join(
            f"#{c['idx']} I={c['diag'].morans_i:+.3f} ({c['diag'].z:+.0f}s)"
            for c in strong)
        lines.append(f"SIGNAL, per structured panel: {detail}.")

    return "\n".join(textwrap.fill(ln, width=width) for ln in lines if ln)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ci.add_input_args(ap)
    ap.add_argument("--out", default=None,
                    help="output PNG (default in --interp-dir)")
    ap.add_argument("--pathways", default=None,
                    help="comma-separated pathway indices to show, in order. "
                         "Default: the top --top-pathways by |IG|.")
    ap.add_argument("--top-pathways", type=int, default=3,
                    help="how many columns when --pathways is not given")
    ap.add_argument("--top-genes", type=int, default=5)
    ap.add_argument("--patches-per-pathway", type=int, default=4)
    ap.add_argument("--patch-px", type=int, default=128,
                    help="edge of each H&E crop in the strip")
    ap.add_argument("--type-of-path", default=ci.env("INTERP_TYPE_OF_PATH", "combine"),
                    help="which <type>_signatures.csv names the pathways "
                         "(combine / hallmarks / xena) — must match the run "
                         "[env: INTERP_TYPE_OF_PATH]")
    ap.add_argument("--metadata-dir", default=ci.env("INTERP_METADATA_DIR"),
                    help="dir holding <type_of_path>_signatures.csv "
                         "(default datasets_csv/metadata)")
    ap.add_argument("--cmap", default=DEFAULT_CMAP,
                    help=f"default '{DEFAULT_CMAP}' to match the published "
                         f"figure; '{hm.DEFAULT_CMAP}' for the perceptually "
                         f"ordered sequential ramp, or any matplotlib name")
    ap.add_argument("--norm", default="rank",
                    choices=("rank", "percentile", "minmax", "signed"))
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Gaussian sigma in tile widths; 0 (default) keeps the "
                         "crisp per-patch mosaic of the paper")
    ap.add_argument("--texture", type=float, default=0.0,
                    help="0..1 — shade tiles by the underlying H&E luminance to "
                         "hint at tissue architecture; 0 = flat, as published")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    if not args.interp_dir:
        raise SystemExit("missing required path: --interp-dir (or INTERP_OUTDIR)")
    if args.label_file and not os.path.isfile(args.label_file):
        raise SystemExit(f"--label-file: no such file: {args.label_file}")

    h5_path, slide_path, slide_id = ci.resolve_files(args)
    cross_attn = ci.load_cross_attn(args.interp_dir, args.case_id, args.cross_attn)

    slide, thumb, coords, ds = ci.open_slide_and_coords(
        h5_path, slide_path, args.downsample)
    print(f"[info] slide '{slide_id}': thumbnail {thumb.shape[1]}x{thumb.shape[0]} "
          f"px, downsample {ds:.1f}, {coords.shape[0]} patches")
    ci.check_patch_alignment(coords, cross_attn, h5_path)

    stride = hm.infer_patch_stride(coords, declared=args.patch_size)

    path_df = ci.load_pathway_importance(args.interp_dir, args.case_id)
    gene_df = ci.load_gene_importance(args.interp_dir, args.case_id)

    if args.pathways:
        tops = [int(s) for s in args.pathways.split(",") if s.strip() != ""]
        bad = [p for p in tops if not 0 <= p < cross_attn.shape[0]]
        if bad:
            raise SystemExit(f"--pathways out of range for a "
                             f"{cross_attn.shape[0]}-pathway model: {bad}")
    else:
        tops = ci.top_pathways(path_df, cross_attn, args.top_pathways)
    print(f"[info] columns: {tops}")

    # Prefer names the run itself recorded; fall back to the signature CSV.
    names = PathwayNames.from_importance_csv(path_df)
    if not names:
        names = PathwayNames.load(type_of_path=args.type_of_path,
                                  metadata_dir=args.metadata_dir)
    names = names.check(cross_attn.shape[0])

    out = args.out or os.path.join(
        args.interp_dir, f"{args.case_id}_cross_modal.png")
    summary = ci.load_summary(args.interp_dir, args.case_id)
    bits = [args.case_id]
    if "risk" in summary:
        bits.append(f"predicted risk {summary['risk']:+.2f}")
    if summary.get("event_time") is not None:
        censored = summary.get("censored")
        label = "last follow-up" if censored else "survival time"
        bits.append(f"{label} {summary['event_time']:.0f} months")
    bits.append(f"{coords.shape[0]} tiles · {stride} px at level 0")

    render(args.case_id, cross_attn, coords, thumb, ds, slide, stride, tops,
           names, gene_df, out, cmap_name=args.cmap, norm_mode=args.norm,
           smooth=args.smooth, texture=args.texture, n_genes=args.top_genes,
           n_patches=args.patches_per_pathway, patch_px=args.patch_px,
           title=args.title, subtitle="  ·  ".join(bits), dpi=args.dpi)
    slide.close()
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
