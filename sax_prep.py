"""sax_prep.py — build the image___/masks___/saxdf___ files the app expects,
from a single SAX image .nii.gz and a single mask .nii.gz (paths below).

All data is read from those .nii.gz files (and the saxdf CSV for the 4CH
overlay). initialize_app() then loads the case via CASE_UID like any other.
"""

import os

import numpy as np
import nibabel as nib
import pandas as pd

CASE_UID = os.environ.get("FORCE_CASE_UID", "sax_case")

# --- single hardcoded input files (edit these paths) ----------------------
SAX_IMAGE_NII = os.environ.get("FORCE_SAX_IMAGE_NII", "/workspaces/Roundel_FORCE-main-test/CIN-000126-1_20230501/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0/artifacts/image___1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0.nii.gz")
SAX_MASK_NII  = os.environ.get("FORCE_SAX_MASK_NII",  "/workspaces/Roundel_FORCE-main-test/CIN-000126-1_20230501/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0/artifacts/masks___1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0.nii.gz")

# The RICH SAX geometry CSV (per-slice orientation/position/pixelspacing/
# thickness/uid). The real pixel spacing, slice thickness and series UID are read
# from here so the app shows the true values instead of hardcoded ones.
SAX_GEOMETRY_CSV = os.environ.get("FORCE_SAX_CSV", "/workspaces/Roundel_FORCE-main-test/CIN-000126-1_20230501/1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0/artifacts/saxdf___1.3.6.1.4.1.5962.99.1.3667672128.100149162.1777489195413.50456.0.csv")

# Fallback spacing/thickness ONLY if the geometry CSV can't be read. The env
# vars override the CSV if you need to force a value.
PIXEL_SPACING = os.environ.get("FORCE_PIXEL_SPACING")        # None -> read from CSV
SLICE_THICKNESS = os.environ.get("FORCE_SLICE_THICKNESS")    # None -> read from CSV
_FALLBACK_PIXEL_SPACING = 1.0
_FALLBACK_SLICE_THICKNESS = 10.0


def _read_geometry_meta():
    """Read (pixelspacing, thickness, uid) from the SAX geometry CSV. Env vars
    override; falls back to sane defaults if the CSV is unavailable."""
    ps = uid = thk = None
    try:
        import pandas as pd
        df = pd.read_csv(SAX_GEOMETRY_CSV)
        if "pixelspacing" in df.columns:
            ps = float(df["pixelspacing"].iloc[0])
        if "thickness" in df.columns:
            thk = float(df["thickness"].iloc[0])
        if "uid" in df.columns:
            uid = str(df["uid"].iloc[0])
    except Exception:
        pass
    if PIXEL_SPACING is not None:
        ps = float(PIXEL_SPACING)
    if SLICE_THICKNESS is not None:
        thk = float(SLICE_THICKNESS)
    return (ps if ps is not None else _FALLBACK_PIXEL_SPACING,
            thk if thk is not None else _FALLBACK_SLICE_THICKNESS,
            uid)


def get_display_uid():
    """The real SAX series UID (from the geometry CSV) for display, falling back
    to CASE_UID if unavailable. CASE_UID stays the internal file key."""
    _ps, _thk, uid = _read_geometry_meta()
    return uid or CASE_UID

# these files are assumed already preprocessed; leave the flip/crop OFF. Set
# SAX_SLICE_FLIP=True or CROP_HW to re-apply the old pipeline steps if needed.
SAX_SLICE_FLIP = False
CROP_CENTER = (150, 100)
CROP_HW = None                 # e.g. (256, 256) to crop; None = no crop
# --------------------------------------------------------------------------


def crop_pad_center_xy(image, center_xy, target_hw=(256, 256),
                       inplane_axes=(0, 1), img_pad_value=None):
    ax0, ax1 = inplane_axes
    A, B = image.shape[ax0], image.shape[ax1]
    Ht, Wt = target_hw
    if img_pad_value is None:
        img_pad_value = float(image.min())
    ca, cb = int(round(center_xy[0])), int(round(center_xy[1]))

    def window(ctr, size, extent):
        s = ctr - size // 2
        lo = max(0, s); hi = min(extent, s + size)
        return lo, hi, lo - s

    a0, a1, ad = window(ca, Ht, A)
    b0, b1, bd = window(cb, Wt, B)
    al, bl = a1 - a0, b1 - b0
    out_shape = list(image.shape); out_shape[ax0] = Ht; out_shape[ax1] = Wt
    out = np.full(out_shape, img_pad_value, dtype=image.dtype)
    src = [slice(None)] * image.ndim; src[ax0] = slice(a0, a0 + al); src[ax1] = slice(b0, b0 + bl)
    dst = [slice(None)] * image.ndim; dst[ax0] = slice(ad, ad + al); dst[ax1] = slice(bd, bd + bl)
    out[tuple(dst)] = image[tuple(src)]
    return out


def _as_4d(vol):
    """(X,Y,S) -> (X,Y,S,1); 4D passes through."""
    vol = np.asarray(vol)
    return vol[..., np.newaxis] if vol.ndim == 3 else vol


def _to_label_map(mask):
    """Accept a label-map (X,Y,S,T) or one-hot (X,Y,S,T,C) and return a
    (X,Y,S,T) integer label map."""
    mask = np.asarray(mask)
    if mask.ndim == 5:                       # one-hot (X,Y,S,T,C)
        return np.argmax(mask, -1).astype(np.uint8)
    if mask.ndim == 4:                       # already a label map (X,Y,S,T)
        return mask.astype(np.uint8)
    if mask.ndim == 3:                       # (X,Y,S) single frame
        return mask.astype(np.uint8)[..., np.newaxis]
    raise ValueError(f"unexpected mask shape {mask.shape}")


def _maybe_flip_crop(image):
    if SAX_SLICE_FLIP:
        image = image[:, :, ::-1, :]
    if CROP_HW is not None:
        image = crop_pad_center_xy(image, CROP_CENTER, CROP_HW, inplane_axes=(0, 1))
    return image


def ensure_case_data(data_path):
    """Write image___/masks___/saxdf___ for the case into data_path.

    Rebuilds when the SOURCE PATHS change (so switching datasets always reloads,
    even if the new files' mtimes are older) OR when a source .nii.gz is newer
    than the built files. Otherwise skips the rebuild so it does NOT run on every
    Streamlit rerun (which caused a big per-click delay)."""
    os.makedirs(data_path, exist_ok=True)
    img_path = os.path.join(data_path, f"image___{CASE_UID}.nii.gz")
    msk_path = os.path.join(data_path, f"masks___{CASE_UID}.nii.gz")
    csv_path = os.path.join(data_path, f"saxdf___{CASE_UID}.csv")
    src_path = os.path.join(data_path, f"source___{CASE_UID}.txt")   # records the sources

    # what the build WOULD come from now
    current_src = f"{SAX_IMAGE_NII}\n{SAX_MASK_NII}\n{SAX_GEOMETRY_CSV}"

    built = all(os.path.exists(p) for p in (img_path, msk_path, csv_path))
    if built:
        # if the recorded sources differ from the current ones -> different
        # dataset, force a rebuild.
        recorded = None
        try:
            with open(src_path) as fh:
                recorded = fh.read()
        except OSError:
            recorded = None

        if recorded == current_src:
            # same dataset: skip only if outputs are newer than the sources
            try:
                newest_src = max(os.path.getmtime(SAX_IMAGE_NII), os.path.getmtime(SAX_MASK_NII))
                oldest_out = min(os.path.getmtime(img_path), os.path.getmtime(msk_path),
                                 os.path.getmtime(csv_path))
                if newest_src <= oldest_out:
                    return CASE_UID            # up to date -> no rebuild
            except OSError:
                return CASE_UID
        # else: sources changed -> fall through and rebuild

    # read the two single .nii.gz files
    image = _as_4d(np.asarray(nib.load(SAX_IMAGE_NII).dataobj)).astype(np.float32)  # (X,Y,S,T)
    labels = _to_label_map(np.asarray(nib.load(SAX_MASK_NII).dataobj))              # (X,Y,S,T)

    image = _maybe_flip_crop(image)
    if SAX_SLICE_FLIP or CROP_HW is not None:
        labels = _maybe_flip_crop(labels)          # keep mask aligned with image

    # align T and S if image/mask differ (use the common min)
    T = min(image.shape[-1], labels.shape[-1])
    image, labels = image[..., :T], labels[..., :T]
    S = min(image.shape[2], labels.shape[2])
    image, labels = image[:, :, :S, :], labels[:, :, :S, :]

    nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), img_path)
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), np.eye(4)), msk_path)
    ps, thk, _uid = _read_geometry_meta()
    pd.DataFrame({"pixelspacing": [ps],
                  "thickness": [thk]}).to_csv(csv_path, index=False)

    # record the sources this build came from (so a dataset switch is detected)
    try:
        with open(src_path, "w") as fh:
            fh.write(current_src)
    except OSError:
        pass

    # a rebuild means the data changed -> the app's cached edited mask would
    # shadow the new volume; remove it so the fresh mask is used. The FORCE app
    # keeps its cache in a 'cache' dir keyed on the series uid (== CASE_UID).
    for cdir in ("cache", os.path.join(os.path.dirname(data_path) or ".", "cache")):
        for stale in (os.path.join(cdir, f"masks___{CASE_UID}.npy"),
                      os.path.join(cdir, f"config___{CASE_UID}.json")):
            try:
                if os.path.exists(stale):
                    os.remove(stale)
            except OSError:
                pass
    return CASE_UID