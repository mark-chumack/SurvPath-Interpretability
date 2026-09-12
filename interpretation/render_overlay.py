"""Draw a per-patch vector onto a slide thumbnail.

``interpret_survpath.py`` produces two kinds of per-patch scalar: pathway ->
patch cross-attention (one vector per pathway) and integrated gradients on the
patch features (one vector per case). Both are painted the same way, on the same
canvas, so the figures sitting side by side in an output folder are comparable.
That painting lives here; the model, the dataset and Captum stay in
``interpret_survpath.py``.

The pixel work itself (tile footprint, normalisation, colour ramp, alpha,
legend, caption) is in ``interpretation.heatmap_utils``, which
``render_gene_pathway_map.py`` also calls — this module is the layer between it
and a slide on disk: resolve the .h5 coords and the WSI, pick a pyramid level,
infer the tile stride, then hand off.

    from interpretation.render_overlay import render_overlay, slide_canvas

It also runs standalone, repainting what ``interpret_survpath.py`` already wrote
— no model, no dataset, no Captum, no GPU — which is what you want when the
numbers are fine and only the look is wrong (ramp, alpha, focus, mosaic vs
overlay). One PNG per pathway, same file names the full run produces:

    python -m interpretation.render_overlay \
        --interp-dir results/results_brca/interpret \
        --case-id    TCGA-AC-A23E \
        --coords-dir /path/to/CLAM/patches_h5 \
        --slide-dir  /path/to/slides \
        --top-pathways 5

Inputs, all from --interp-dir: <case>_cross_attn_pathways.npy (required),
<case>_pathway_importance.csv (optional — ranks the pathways; without it they
are ranked by mean attention) and, for --ig, <case>_ig_patch_attr.npy.
See ``scripts/render_overlay.sh`` for a filled-in invocation.
"""

import numpy as np
if not hasattr(np, "typeDict"):          # removed in NumPy 1.24; h5py 2.x still uses it
    np.typeDict = np.sctypeDict

from interpretation.heatmap_utils import (
    SLIDE_EXTS as _SLIDE_EXTS,
    find_case_file as _find,
    ATTENTION_SEMANTICS,
)


def slide_canvas(slide_id, coords_dir, slide_dir, downsample, n_patches,
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


def paint_and_save(values, coords, thumb, ds, stride, out_path, title, subject,
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

    canvas = slide_canvas(slide_id, coords_dir, slide_dir, downsample,
                          attn_vec.shape[0], patch_size)
    if canvas is None:
        return False
    coords, thumb, ds, stride = canvas

    subject = "this pathway" if pathway_idx is None else f"pathway {pathway_idx}"
    title = (f"{slide_id} — where {subject} looked"
             if pathway_idx is not None else f"{slide_id} — pathway attention")
    return paint_and_save(attn_vec, coords, thumb, ds, stride, out_path, title,
                          subject, style, ATTENTION_SEMANTICS,
                          label=f"pathway {pathway_idx}")
