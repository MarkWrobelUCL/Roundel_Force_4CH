"""align_4ch.py — draw a SAX slice's location as a cross-reference line on a 4CH
image. Geometry follows the Force_align notebook: intersect the SAX slice plane
with the 4CH image plane and render the intersection (plus +/- half-thickness).

The 4CH and SAX DICOM folders are read from disk at runtime (they sit next to the
app for now). Only header geometry is needed from the SAX side; the 4CH pixels are
shown as the background.
"""

import base64
import glob
import io
import os

import numpy as np


# ------------------------------------------------------------- streamlit-safe helpers
def _session_get(key, default=None):
    """Read st.session_state[key] defensively. Returns default if Streamlit's
    runtime/session isn't ready yet (avoids 'Tried to use SessionInfo before it
    was initialized' during early reruns / cached execution / threads)."""
    try:
        import streamlit as st
        return st.session_state.get(key, default)
    except Exception:
        return default


def _st_cache_data_safe():
    """Return st.cache_data as a decorator if available, else a no-op decorator
    (so the module imports and runs even when the Streamlit runtime isn't ready)."""
    try:
        import streamlit as st
        return st.cache_data(show_spinner=False)
    except Exception:
        def _noop(fn):
            return fn
        return _noop


# ------------------------------------------------------------------ geometry
def _plane(ds):
    iop = np.asarray(ds.ImageOrientationPatient, float)
    row_dir, col_dir = iop[:3], iop[3:]
    origin = np.asarray(ds.ImagePositionPatient, float)
    ps = np.asarray(ds.PixelSpacing, float)              # [row_spacing, col_spacing]
    n = np.cross(row_dir, col_dir)
    n /= np.linalg.norm(n)
    return origin, row_dir, col_dir, n, ps


def _to_pixel(P, origin, row_dir, col_dir, ps):
    d = P - origin
    return np.dot(d, row_dir) / ps[1], np.dot(d, col_dir) / ps[0]   # (col, row)


def _line_in_pixels(plane_pt, plane_n, o4, r4, c4, n4, ps4):
    d = np.cross(n4, plane_n)
    if np.linalg.norm(d) < 1e-8:
        return None
    d /= np.linalg.norm(d)
    p0 = np.linalg.solve(np.array([n4, plane_n, d]),
                         np.array([n4 @ o4, plane_n @ plane_pt, 0.0]))
    c0, r0 = _to_pixel(p0,     o4, r4, c4, ps4)
    c1, r1 = _to_pixel(p0 + d, o4, r4, c4, ps4)
    return np.array([c0, r0]), np.array([c1 - c0, r1 - r0])


def _clip(p, v, cols, rows):
    px, py = p
    vx, vy = v
    t0, t1 = -np.inf, np.inf
    for pk, qk in [(-vx, px), (vx, cols - 1 - px), (-vy, py), (vy, rows - 1 - py)]:
        if abs(pk) < 1e-12:
            if qk < 0:
                return None
        else:
            t = qk / pk
            if pk < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return p + t0 * v, p + t1 * v


# ------------------------------------------------------------------ dicom I/O
def _read_series(folder):
    """Return DICOM datasets in a folder (headers only), one per file."""
    import pydicom
    dsets = []
    for path in sorted(glob.glob(os.path.join(folder, "*.dcm"))):
        try:
            dsets.append(pydicom.dcmread(path, force=True))
        except Exception:
            continue
    return dsets


def _sax_slice_positions(sax_dsets):
    """Group SAX datasets by unique slice location (ignoring cine frames) and
    return them ordered along the slice normal: list of (position_key, dataset)."""
    if not sax_dsets:
        return []
    iop = np.asarray(sax_dsets[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    normal /= np.linalg.norm(normal)

    by_pos = {}
    for ds in sax_dsets:
        ipp = np.asarray(ds.ImagePositionPatient, float)
        key = round(float(np.dot(ipp, normal)), 3)        # position along normal
        by_pos.setdefault(key, ds)                        # first frame per location
    return sorted(by_pos.items(), key=lambda kv: kv[0])


# ------------------------------------------------------------------ render
def _window(ds):
    """Return a display-normalised float image in [0,1], or None if the pixel
    data can't be decoded (e.g. JPEG-compressed DICOM with a broken decoder).

    Uses a robust percentile stretch by default (adapts to the actual pixel
    distribution, so the 4CH shows with good contrast). The stored DICOM
    WindowCenter/WindowWidth are often tuned for a different context and can look
    very dark, so they are NOT used unless USE_STORED_WINDOW is enabled."""
    try:
        img = ds.pixel_array.astype(float)
    except Exception:
        return None

    if USE_STORED_WINDOW and "WindowCenter" in ds and "WindowWidth" in ds:
        wc = float(np.ravel(ds.WindowCenter)[0])
        ww = float(np.ravel(ds.WindowWidth)[0])
        if ww > 0:
            return np.clip((img - (wc - ww / 2)) / ww, 0, 1)

    # robust, data-adaptive display stretch
    lo, hi = np.percentile(img, [WINDOW_PCT_LOW, WINDOW_PCT_HIGH])
    if hi > lo:
        return np.clip((img - lo) / (hi - lo), 0, 1)
    # flat image -> normalise by max so it isn't all-black
    m = float(img.max())
    return img / m if m > 0 else img


def _dir_sig(folder):
    """Cheap signature so the cache invalidates if the folder changes."""
    try:
        files = sorted(glob.glob(os.path.join(folder, "*")))
        return (folder, len(files), max((os.path.getmtime(f) for f in files), default=0))
    except Exception:
        return (folder, 0, 0)


def _build_align_bundle(fourch_dir, sax_csv, _sig4, _sigS, loc_index=None):
    """Precompute everything that is CONSTANT across slice/phase changes:
      * the ordered, windowed 4CH backgrounds as one uint8 array (T, H, W, 3)
      * the 4CH plane params + image size
      * each SAX slice's (origin, normal, thickness), ordered along the normal
    The per-interaction work (which frame, which lines) is then trivial.

    Cached via render_4ch_view's st.cache_data wrapper (keyed on the sigs +
    loc_index). SAX geometry comes from the saxdf CSV; 4CH from its DICOM cine.
    loc_index selects which 4CH spatial slice to display.
    """
    series = _read_fourch_ordered(fourch_dir, loc_index=loc_index)
    sax_pos = _sax_slices_from_csv(sax_csv) if os.path.isfile(sax_csv) else []
    if not series or not sax_pos:
        return None

    ds4 = series[0]
    o4, r4, c4, n4, ps4 = _plane(ds4)
    rows, cols = int(ds4.Rows), int(ds4.Columns)

    # window every 4CH frame once -> (T, H, W, 3) uint8; also capture EACH frame's
    # plane geometry, because 4CH cine frames can have per-frame ImagePositionPatient
    # (the heart/table shifts between phases). Using one frame's plane for all
    # phases makes the line drift when the background translates (e.g. systole).
    bgs = []
    planes4 = []
    for ds in series:
        img = _window(ds)
        if img is None:
            img = np.zeros((rows, cols), dtype=float)
        bgs.append((np.stack([img] * 3, -1) * 255).astype(np.uint8))
        planes4.append(_plane(ds))                     # (o4, r4, c4, n4, ps4) per frame
    backgrounds = np.stack(bgs, axis=0)

    # SAX slice planes (origin, normal, thickness)
    sax_planes = []
    for _key, dss in sax_pos:
        oS, _, _, nS, _ = _plane(dss)
        thk = float(getattr(dss, "SliceThickness", 8.0))
        sax_planes.append((oS, nS, thk))

    return {
        "backgrounds": backgrounds,           # (T, H, W, 3)
        "plane4": planes4[0],                 # kept for compatibility
        "planes4": planes4,                   # per-frame 4CH plane params
        "rows": rows, "cols": cols,
        "sax_planes": sax_planes,             # list of (origin, normal, thk)
        "n_4ch": len(series),
        "n_slices": len(sax_planes),
    }


def _fourch_location_groups(fourch_dir):
    """Group 4CH DICOMs by spatial location. Returns (ordered_loc_keys, groups)
    where groups maps loc_key -> list of datasets. Each loc_key is a rounded
    ImagePositionPatient tuple (or None). ordered_loc_keys is sorted by position.

    Some '4CH' acquisitions contain more than one spatial slice interleaved across
    the cardiac cycle; grouping by position lets the user pick which slice to view
    (and keeps phase-picking within one slice so the background doesn't translate
    between diastole and systole)."""
    series = _read_series(fourch_dir)
    if not series:
        return [], {}

    def loc_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None:
            return None
        return tuple(round(float(x), 1) for x in ipp)     # group by position (0.1mm)

    groups = {}
    for ds in series:
        groups.setdefault(loc_key(ds), []).append(ds)
    ordered = sorted(groups.keys(), key=lambda k: (k is None, k))
    return ordered, groups


def _phase_order(frames):
    """Order a location's frames by cardiac phase (TriggerTime, else InstanceNumber)."""
    def phase_key(ds):
        tt = getattr(ds, "TriggerTime", None)
        return (0, float(tt)) if tt is not None else (1, int(getattr(ds, "InstanceNumber", 0) or 0))
    return sorted(frames, key=phase_key)


def _default_loc_index(ordered_locs, groups):
    """Which location to show by default: FORCE_4CH_LOC_INDEX if set & valid,
    else the one with the most frames (the main cine)."""
    idx = int(os.environ.get("FORCE_4CH_LOC_INDEX", "-1"))
    if 0 <= idx < len(ordered_locs):
        return idx
    if not ordered_locs:
        return 0
    best = max(ordered_locs, key=lambda k: len(groups[k]))
    return ordered_locs.index(best)


def _read_fourch_ordered(fourch_dir, loc_index=None):
    """Return the 4CH cine frames for ONE spatial slice, ordered by cardiac phase.
    loc_index picks which location (0-based, by position order); None -> default
    (most frames, or FORCE_4CH_LOC_INDEX)."""
    ordered, groups = _fourch_location_groups(fourch_dir)
    if not ordered:
        return []
    if loc_index is None or not (0 <= loc_index < len(ordered)):
        loc_index = _default_loc_index(ordered, groups)
    return _phase_order(groups[ordered[loc_index]])


def _lines_for_slice(bundle, slice_idx):
    """Cheap per-slice work: the 3 cross-reference line segments on the 4CH."""
    o4, r4, c4, n4, ps4 = bundle["plane4"]
    rows, cols = bundle["rows"], bundle["cols"]
    planes = bundle["sax_planes"]
    if not planes:
        return []
    slice_idx = int(np.clip(slice_idx, 0, len(planes) - 1))
    oS, nS, thk = planes[slice_idx]
    plan = [(oS, (0, 230, 0), "solid"),
            (oS + (thk / 2) * nS, (0, 200, 255), "solid"),
            (oS - (thk / 2) * nS, (0, 200, 255), "solid")]
    lines = []
    for pt, color, style in plan:
        res = _line_in_pixels(pt, nS, o4, r4, c4, n4, ps4)
        if res is None:
            continue
        seg = _clip(*res, cols, rows)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        lines.append({"seg": ((float(x0), float(y0)), (float(x1), float(y1))),
                      "color": color, "style": style})
    return lines


def sax_lines_on_4ch(fourch_series, sax_positions, slice_idx,
                     sax_frame=0, sax_nframes=1, thickness_mm=None):
    """Return (rgb_uint8, lines) for the 4CH background with the chosen SAX slice
    drawn on it. `lines` is a list of dicts {seg:((x0,y0),(x1,y1)), color, style}.

    fourch_series : ordered 4CH cine (list of Datasets); geometry is shared, only
                    the pixels differ between frames.
    sax_positions : list of (position_key, dataset) for the SAX slices, ordered
                    along the slice normal (as returned by _resolve_sax_slices).
    slice_idx     : which SAX slice -- same index FORCE's slice slider uses.
    sax_frame     : the SAX cardiac timepoint being viewed (e.g. dia_idx/sys_idx).
    sax_nframes   : total SAX cine frames, used to map phase -> nearest 4CH frame.

    The 4CH background is picked as the frame whose cardiac phase is NEAREST the
    SAX phase (sax_frame / sax_nframes), so different SAX/4CH frame counts are
    handled. Source-agnostic: never touches folders or session state.
    """
    if isinstance(fourch_series, (list, tuple)):
        series = list(fourch_series)
    else:
        series = [fourch_series]                       # tolerate a single dataset
    series = [d for d in series if d is not None]
    if not series:
        raise ValueError("No 4CH dataset available")

    # geometry is identical across cine frames -> take it from the first
    ds4 = series[0]
    o4, r4, c4, n4, ps4 = _plane(ds4)
    rows, cols = int(ds4.Rows), int(ds4.Columns)

    # phase-match: pick the 4CH frame nearest the SAX phase fraction
    n4f = len(series)
    if n4f > 1 and sax_nframes > 0:
        phase = float(sax_frame) / float(sax_nframes)          # 0..1
        j = int(round(phase * n4f)) % n4f                      # nearest 4CH frame
    else:
        j = 0
    img = _window(series[j])
    if img is None:
        img = np.zeros((rows, cols), dtype=float)     # decoder failed -> blank bg
    rgb = (np.stack([img] * 3, -1) * 255).astype(np.uint8)

    if not sax_positions:
        return rgb, []
    slice_idx = int(np.clip(slice_idx, 0, len(sax_positions) - 1))
    dss = sax_positions[slice_idx][1]
    oS, _, _, nS, _ = _plane(dss)
    thk = float(thickness_mm if thickness_mm is not None
                else getattr(dss, "SliceThickness", 8.0))

    plan = [(oS,               (0, 230, 0), "solid"),
            (oS + (thk / 2) * nS, (0, 200, 255), "solid"),
            (oS - (thk / 2) * nS, (0, 200, 255), "solid")]
    lines = []
    for pt, color, style in plan:
        res = _line_in_pixels(pt, nS, o4, r4, c4, n4, ps4)
        if res is None:
            continue
        seg = _clip(*res, cols, rows)
        if seg is None:
            continue
        (x0, y0), (x1, y1) = seg
        lines.append({"seg": ((float(x0), float(y0)), (float(x1), float(y1))),
                      "color": color, "style": style})
    return rgb, lines


def n_sax_slices(case=None):
    """Number of SAX slices available for the current case."""
    return len(_resolve_sax_slices(case))


# ==========================================================================
# SOURCE RESOLUTION  --  the ONE place to change when the real data arrives.
# ==========================================================================
# Everything downstream (geometry, line drawing, the Streamlit view) is generic.
# It only needs two things per case:
#   * the 4CH dataset      (a pydicom Dataset: header geometry + pixels)
#   * the SAX slice datasets, ordered along the slice normal (headers only)
#
# `_resolve_sources()` is the seam. Today it reads two hardcoded folders that sit
# next to the app. To wire in the real data, replace ONLY the body of the two
# `_resolve_*` functions below -- e.g. look the 4CH/SAX up by the current
# `sax_series_uid`, or read geometry FORCE already loaded. Nothing else changes.
# --------------------------------------------------------------------------

# --- 4CH cine: a folder of DICOMs (the background image) ------------------
# Hardcoded for this case; override with FORCE_4CH_DIR.
FOURCH_DIR = os.environ.get("FORCE_4CH_DIR", "/workspaces/Roundel_FORCE-main-test/CIN-000126-1_20230501/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.51064.0")

# --- SAX slice geometry: read from the saxdf CSV (DICOMs unavailable) -------
# This must be the RICH saxdf CSV that has per-slice `orientation` and
# `position` columns (exported from the original SAX series), NOT the minimal
# pixelspacing/thickness CSV that sax_prep writes for metrics.
SAX_CSV = os.environ.get("FORCE_SAX_CSV", "/workspaces/Roundel_FORCE-main-test/CIN-000126-1_20230501/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0/artifacts/saxdf___1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0.csv")

# --- 4CH display normalisation --------------------------------------------
# Percentile stretch bounds for display contrast (lower/upper). Widen toward
# 0/100 for a flatter stretch, narrow (e.g. 2/98) for punchier contrast.
WINDOW_PCT_LOW = float(os.environ.get("FORCE_WINDOW_PCT_LOW", 1.0))
WINDOW_PCT_HIGH = float(os.environ.get("FORCE_WINDOW_PCT_HIGH", 99.0))
# Use the DICOM's stored WindowCenter/WindowWidth instead of the percentile
# stretch. Off by default because stored windows often render very dark.
USE_STORED_WINDOW = os.environ.get("FORCE_USE_STORED_WINDOW", "0").lower() in ("1", "true", "yes")


def _image_slice_flip():
    """Whether the SAX IMAGE volume was built with a slice flip. Read straight
    from sax_prep so the overlay's slice order ALWAYS matches the displayed
    image — no manual setting to keep in sync. Falls back to False if sax_prep
    isn't importable."""
    try:
        import sax_prep
        return bool(getattr(sax_prep, "SAX_SLICE_FLIP", False))
    except Exception:
        return False


class _SaxRow:
    """Lightweight stand-in for a pydicom Dataset, exposing only the geometry
    attributes the alignment math reads, populated from a saxdf CSV row."""
    __slots__ = ("ImageOrientationPatient", "ImagePositionPatient",
                 "PixelSpacing", "SliceThickness")

    def __init__(self, iop, ipp, ps, thk):
        self.ImageOrientationPatient = iop
        self.ImagePositionPatient = ipp
        self.PixelSpacing = ps
        self.SliceThickness = thk


def _parse_vec(s):
    """Parse a '[a, b, c]' string (or list) into a list of floats."""
    import ast
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]
    return [float(x) for x in ast.literal_eval(str(s))]


def _sax_slices_from_csv(csv_path):
    """Read a saxdf___*.csv and return SAX slice geometry as
    list of (position_key, _SaxRow), ordered to match the app's slice slider.
    One entry per unique slice (cine frames de-duplicated via slicelocation).

    Ordering uses the CSV `slicelocation` column (falling back to position-on-
    normal), matching how the SAX image stack is assembled, then applies the same
    slice flip the image volume was built with (read from sax_prep) so
    display-slice d lines up with the correct physical slice — fully automatic.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    iop0 = _parse_vec(df["orientation"].iloc[0])
    normal = np.cross(iop0[:3], iop0[3:])
    normal /= np.linalg.norm(normal)

    have_sl = "slicelocation" in df.columns

    by_key = {}
    for _, r in df.iterrows():
        ipp = _parse_vec(r["position"])
        iop = _parse_vec(r["orientation"])
        if have_sl and not pd.isna(r.get("slicelocation")):
            key = round(float(r["slicelocation"]), 3)          # order by slicelocation
        else:
            key = round(float(np.dot(np.asarray(ipp), normal)), 3)
        if key in by_key:
            continue                                           # first frame per slice
        ps = float(r["pixelspacing"]) if "pixelspacing" in df.columns else 1.0
        thk = float(r["thickness"]) if "thickness" in df.columns else 8.0
        by_key[key] = _SaxRow(iop, ipp, [ps, ps], thk)

    ordered = sorted(by_key.items(), key=lambda kv: kv[0])     # slicelocation ascending
    # match the SAME slice flip the image volume was built with, so display-slice
    # d lines up with the correct physical slice (auto-derived, no manual flag).
    if _image_slice_flip():
        ordered = ordered[::-1]
    return ordered


def _resolve_fourch(case=None):
    """Return the 4CH pydicom Dataset for this case (geometry + first-frame
    pixels), or None.

    PLACEHOLDER: reads the first DICOM in the hardcoded 4CH folder. Replace the
    body to select the real per-case 4CH series (e.g. keyed off `case`).
    """
    series = _resolve_fourch_series(case)
    return series[0] if series else None


def _resolve_fourch_series(case=None):
    """Return the 4CH cine for ONE spatial slice, ordered by cardiac phase.
    Delegates to _read_fourch_ordered so location-grouping (systole/diastole on
    the correct slice) is applied consistently.

    PLACEHOLDER: reads the hardcoded 4CH folder. Replace alongside _resolve_fourch
    when wiring the real per-case 4CH.
    """
    fourch_dir = _session_get("force_4ch_dir")
    fourch_dir = fourch_dir or FOURCH_DIR
    if not os.path.isdir(fourch_dir):
        return []
    return _read_fourch_ordered(fourch_dir)


def _resolve_sax_slices(case=None):
    """Return the SAX slice geometry ordered along the slice normal:
    list of (position_key, _SaxRow). Read from the saxdf CSV (the SAX DICOMs
    are no longer available). Replace the CSV path resolution here for real
    per-case wiring.
    """
    sax_csv = _session_get("force_sax_csv")
    sax_csv = sax_csv or SAX_CSV
    if not os.path.isfile(sax_csv):
        return []
    return _sax_slices_from_csv(sax_csv)


def _resolve_sources(case=None):
    """Bundle the resolvers. Returns (fourch_series, sax_positions), where
    fourch_series is the ordered cine (list of Datasets) for phase syncing."""
    return _resolve_fourch_series(case), _resolve_sax_slices(case)


# ------------------------------------------------------------------ draw
def _draw_lines(rgb, lines, width=1):
    """Draw the cross-reference lines onto the 4CH rgb (uint8). Anti-aliased via
    a supersampled overlay so tilted lines look clean."""
    from PIL import Image, ImageDraw
    ss = 3
    H, W = rgb.shape[:2]
    base = Image.fromarray(rgb).convert("RGBA").resize((W * ss, H * ss), Image.NEAREST)
    ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for ln in lines:
        (x0, y0), (x1, y1) = ln["seg"]
        col = tuple(ln["color"]) + (230,)
        if ln["style"] == "dash":
            # dashed: sample along the segment
            x0s, y0s, x1s, y1s = x0 * ss, y0 * ss, x1 * ss, y1 * ss
            length = ((x1s - x0s) ** 2 + (y1s - y0s) ** 2) ** 0.5
            steps = max(1, int(length / (10 * ss)))
            for k in range(steps):
                if k % 2:
                    continue
                a = k / steps
                b = (k + 1) / steps
                dr.line([(x0s + (x1s - x0s) * a, y0s + (y1s - y0s) * a),
                         (x0s + (x1s - x0s) * b, y0s + (y1s - y0s) * b)],
                        fill=col, width=width * ss)
        else:
            dr.line([(x0 * ss, y0 * ss), (x1 * ss, y1 * ss)], fill=col, width=width * ss)
    out = Image.alpha_composite(base, ov).resize((W, H), Image.LANCZOS).convert("RGB")
    return np.array(out)


def render_4ch_view(slice_idx, sax_frame=0, sax_nframes=1, case=None, disp_w=450):
    """Streamlit widget: show the 4CH image with the current SAX slice drawn on it,
    with the 4CH background synced to the SAX cardiac phase.

    The heavy work (reading/decoding the DICOM folders, windowing every 4CH frame,
    computing SAX slice planes) is cached once per dataset. Changing slice or phase
    then only recomputes 3 line segments + the overlay draw -> near-instant.
    """
    import streamlit as st

    fourch_dir, sax_csv = _resolve_dirs(case)
    if not (fourch_dir and sax_csv and os.path.isdir(fourch_dir) and os.path.isfile(sax_csv)):
        st.info("4CH / SAX geometry not available for this case yet.\n\n"
                "The 4CH cine is read from a DICOM folder (`FORCE_4CH_DIR`) and the "
                "SAX slice geometry from a saxdf CSV (`FORCE_SAX_CSV`). Set those, "
                "or wire the real per-case lookup into `_resolve_dirs`.")
        return

    # --- 4CH spatial-slice selection -------------------------------------
    # Some 4CH acquisitions contain multiple spatial slices interleaved across
    # the cardiac cycle. Offer a dropdown to pick which one; if there's only a
    # single slice, no dropdown is shown. The dropdown WIDGET is rendered below
    # the image (further down), but we read its current value now so the bundle
    # uses the chosen slice; the value persists in session_state across reruns.
    loc_labels = _fourch_location_labels(fourch_dir)
    multi_slice = len(loc_labels) > 1
    loc_index = None
    if multi_slice:
        _ordered, _sizes, _labels, default_i = _fourch_location_meta(
            fourch_dir, _dir_sig(fourch_dir))
        sel = _session_get("fourch_loc_index", None)
        loc_index = int(sel) if isinstance(sel, int) and 0 <= sel < len(loc_labels) else default_i
    # (single slice -> loc_index stays None -> the one slice is used, no dropdown)

    bundle = _cached_bundle(fourch_dir, sax_csv, loc_index)
    if bundle is None:
        st.warning("Could not build the 4CH overlay (check the 4CH folder and SAX CSV).")
        return

    # --- per-interaction: pick phase-matched frame + compute lines (cheap) ---
    n4 = bundle["n_4ch"]
    if n4 > 1 and sax_nframes > 0:
        j = int(round((float(sax_frame) / float(sax_nframes)) * n4)) % n4
    else:
        j = 0
    rgb = bundle["backgrounds"][j]                     # cached windowed frame
    lines = _lines_for_slice(bundle, slice_idx)

    if rgb.max() == 0:
        st.caption("⚠️ 4CH image pixels could not be decoded (compressed DICOM); "
                   "showing the slice line on a blank background.")
    if not lines:
        st.caption("This SAX slice does not intersect the 4CH field of view.")
    rgb = _draw_lines(rgb.copy(), lines)
    # Render as an inline base64 data URI instead of st.image(). st.image caches
    # each frame as a temp media file; rapid slider changes evict them before the
    # browser fetches, producing harmless but noisy "MediaFileHandler: Missing
    # file" logs. A data URI embeds the bytes directly -> no temp file, no misses.
    _show_image_datauri(rgb, disp_w)
    st.caption("SAX slice location on the 4CH view — "
               "**green** = slice centre, **cyan** = slice edges (± half-thickness). ")

    # --- 4CH slice dropdown, placed BELOW the image + caption -------------
    if multi_slice:
        options = [i for i, _lbl in loc_labels]
        # If the widget's key already exists in session_state, let it own the
        # value (don't also pass index, which Streamlit warns about). On first
        # render, seed it with the default index.
        kwargs = {}
        if _session_get("fourch_loc_index", None) is None:
            kwargs["index"] = loc_index if loc_index is not None else 0
        st.selectbox(
            "4CH slice",
            options,
            format_func=lambda i: dict(loc_labels)[i],
            key="fourch_loc_index",
            help="This 4CH series has more than one spatial slice. Pick which one "
                 "to display behind the SAX line.",
            **kwargs,
        )


def _show_image_datauri(rgb, width):
    """Display an RGB uint8 array inline via a base64 PNG data URI (bypasses
    Streamlit's media-file cache)."""
    import streamlit as st
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="width:{int(width)}px;max-width:100%;height:auto;" />',
        unsafe_allow_html=True,
    )


def _resolve_dirs(case=None):
    """The 4CH folder and SAX CSV for this case. PLACEHOLDER: fixed paths / env /
    session_state. Replace this one function to wire in real per-case lookup."""
    fourch_dir = _session_get("force_4ch_dir")
    sax_csv = _session_get("force_sax_csv")
    return (fourch_dir or FOURCH_DIR), (sax_csv or SAX_CSV)


def _file_sig(path):
    """Signature so the cache invalidates when a file (CSV) changes."""
    try:
        return (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return (path, 0, 0)


@_st_cache_data_safe()
def _cached_build(fdir, scsv, sig4, sigS, loc_index):
    return _build_align_bundle(fdir, scsv, sig4, sigS, loc_index=loc_index)


def _cached_bundle(fourch_dir, sax_csv, loc_index=None):
    """Cached wrapper around _build_align_bundle, keyed on signatures + loc_index
    so it rebuilds when the 4CH folder, the SAX CSV, or the chosen 4CH slice
    changes."""
    return _cached_build(fourch_dir, sax_csv, _dir_sig(fourch_dir),
                         _file_sig(sax_csv), loc_index)


@_st_cache_data_safe()
def _fourch_location_meta(fourch_dir, _sig):
    """Cached: (ordered_loc_keys, group_sizes, labels, default_index) for the 4CH
    folder. Computed once per folder (keyed on _sig) so moving the SAX slider
    doesn't re-read every 4CH DICOM header each rerun. Returns plain data (not the
    dataset lists) so it's cache-friendly."""
    ordered, groups = _fourch_location_groups(fourch_dir)
    sizes = {k: len(v) for k, v in groups.items()}
    labels = []
    for i, k in enumerate(ordered):
        n = sizes[k]
        if k is None:
            labels.append((i, f"Slice {i + 1} (no position, {n} frames)"))
        else:
            labels.append((i, f"Slice {i + 1}  (z={k[2]:.1f} mm, {n} frames)"))
    default_i = _default_loc_index(ordered, groups)
    return ordered, sizes, labels, default_i


def _fourch_location_labels(fourch_dir):
    """Human-readable labels for each 4CH location, for the slice dropdown.
    Returns list of (index, label) ordered by position. Cached per folder."""
    _ordered, _sizes, labels, _default = _fourch_location_meta(fourch_dir, _dir_sig(fourch_dir))
    return labels