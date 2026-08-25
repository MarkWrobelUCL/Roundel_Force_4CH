import os, sys
import glob
import math
import hashlib
import shutil
from pathlib import Path
import io
import json
import base64
from pathlib import Path
import nibabel as nib
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageSequence, ImageDraw, ImageFont
from cv2 import resize, INTER_NEAREST
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from skimage.measure import label as cc_label, regionprops
from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    binary_closing,
    gaussian_filter
)  
from skimage.morphology import disk  
from skimage.measure import find_contours, marching_cubes
import pandas as pd
import time
import cv2

blank_gif_path = f'results/temp/blank.gif'
full_edited_gif_path = f'results/temp/edited.gif'
preprocessed_gif_path = f'results/temp/preprocessed.gif'
edv_esv_gif_path = f'results/temp/edv_esv.gif'
edited_gif_path = f'results/temp/edited_edv_esv.gif'
raw_curve_path = f'results/temp/raw_metrics.png'
edited_curve_path = f'results/temp/edited_metrics.png'
review_list_path = 'results/review.csv'
cache_dir = 'cache'

os.makedirs('results/temp', exist_ok=True)
os.makedirs('results/gifs', exist_ok=True)
os.makedirs('results/masks', exist_ok=True)
os.makedirs('results/edited_sax_df', exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

GIF_W = 150
DISPLAY_W = 400
BACKGROUND_COLOR = (150, 150, 150, 0)
LV_MYO_COLOR = (0, 255, 255, 50) # Blue
LV_COLOR = (255, 10, 10, 50)      # Red

background_idx = 0
lv_myo_idx = 1
lv_idx = 2

channels = [lv_myo_idx, lv_idx]

BRUSH_LABELS = {
    lv_myo_idx: 'Myocardium 🔵',
    lv_idx: 'Blood Pool 🔴',
}

OVERLAY_COLORS = {
    background_idx: BACKGROUND_COLOR,
    lv_myo_idx: LV_MYO_COLOR,
    lv_idx: LV_COLOR,
}

def get_sax_series_uid_list(data_path):
    sax_series_uid_list = sorted([uid.replace('image___','').split('/')[-1].replace('.nii.gz','') for uid in glob.glob(f'{data_path}/*') if 'image' in uid])
    saved_sax_series_uid_list = [uid.split('/')[-1].replace('.csv','') for uid in glob.glob(f'results/edited_sax_df/*')]
    sax_series_uid_list = sorted(set(sax_series_uid_list).difference(set(saved_sax_series_uid_list)))
    return sax_series_uid_list

def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)
    
def save_cached_mask(mask, save_path):
    np.save(save_path, mask)

def load_cached_mask(save_path):
    return np.load(save_path)

def save_mask(mask, save_path):
    nib_mask = nib.Nifti1Image(mask, affine=np.eye(4), dtype='uint8')
    nib.save(nib_mask, save_path)


def download_review_csv(review_list_path):
    available_cases = get_sax_series_uid_list(st.session_state['data_path'])  # needs to be the list of sax_series_uid left in the list, so the list doesn't have ones that Nickie has already completed

    if os.path.exists(review_list_path):
        df = pd.read_csv(review_list_path)
        df = df[df["sax_series_uid"].isin(available_cases)]
    else:
        df = pd.DataFrame(columns=["patient", "study_date", "sax_series_uid"])

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download Review List",data=csv_bytes,file_name=os.path.basename(review_list_path),mime="text/csv",use_container_width=True, icon=":material/download:")


def read_or_create_review_csv(review_list_path, patient, study_date, sax_series_uid):
    if st.button('Mark for Review 📋', type = 'primary', use_container_width=True):
        available_cases = get_sax_series_uid_list(st.session_state["data_path"])
        added_to_list = sax_series_uid in available_cases

        if os.path.exists(review_list_path):
            df = pd.read_csv(review_list_path)
            df = df[df["sax_series_uid"].isin(available_cases)]

            if added_to_list and sax_series_uid not in df["sax_series_uid"].values:
                new_row = {
                    "patient": patient,
                    "study_date": study_date,
                    "sax_series_uid": sax_series_uid,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            if added_to_list:
                df = pd.DataFrame(
                    {
                        "patient": [patient],
                        "study_date": [study_date],
                        "sax_series_uid": [sax_series_uid],
                    }
                )
            else:
                df = pd.DataFrame(columns=["patient", "study_date", "sax_series_uid"])

        df.to_csv(review_list_path, index=False)
        st.rerun()

def load_font(size):
    # Try Linux font
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        pass
    # Try Windows font
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except:
        pass
    # Fallback (non scalable)
    return ImageFont.load_default()

# --------------------------------------------------------------
# Initialization
# --------------------------------------------------------------
def initialize_app(data_path, patient, study_date, sax_series_uid, preprocess=True):
    st.session_state['data_path'] = data_path
    st.session_state['sax_series_uid'] = sax_series_uid
    st.session_state['patient'] = patient
    st.session_state['study_date'] = study_date
    
    # Store the last selected UID in session_state
    if "last_sax_uid" not in st.session_state:
        st.session_state.last_sax_uid = None

    # If user changes series UID, clear relevant session state
    if st.session_state.last_sax_uid != sax_series_uid:
        keys_to_clear = [
            "preprocessed",
            "edited_mask",
            "mask_hash",
            "edv_esv_selected",
            "slice_idx",
            "initialized_all",
            # any other series-specific keys
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.last_sax_uid = sax_series_uid

    if "initialized_all" in st.session_state:
        return

    raw_image = load_nii(f'{data_path}/image___{sax_series_uid}.nii.gz')
    raw_mask = load_nii(f'{data_path}/masks___{sax_series_uid}.nii.gz').astype('uint8')

    sax_df = pd.read_csv(f'{data_path}/saxdf___{sax_series_uid}.csv')

    pixelspacing, thickness = float(sax_df['pixelspacing'].iloc[0]), float(sax_df['thickness'].iloc[0])

    N = len(np.unique(raw_mask))
    raw_mask = np.eye(N, dtype=np.uint8)[raw_mask]
    raw_shape = raw_image.shape

    # -----------------------------
    # Compute raw indices
    # -----------------------------
    volume = np.sum(raw_mask[...,-1], axis=(0,1,2))
    raw_dia_idx = int(np.argmax(volume))
    raw_sys_idx = np.where(volume != 0)[0][np.argmin(volume[volume != 0])]


    # Compute raw metrics
    raw_volume, raw_masses, raw_edv, raw_esv, raw_sv, raw_ef, raw_mass = calculate_sax_metrics(
        raw_mask, pixelspacing, thickness, raw_dia_idx, raw_sys_idx
    )


    st.session_state.raw = {
        "image": raw_image,
        "mask": raw_mask,
        "shape": raw_shape,
        "raw_dia_idx": raw_dia_idx,
        "raw_sys_idx": raw_sys_idx,
        "raw_edv":raw_edv,
        "raw_esv":raw_esv,
        "raw_sv":raw_sv,
        "raw_ef":raw_ef,
        "raw_mass":raw_mass,
        "raw_volume": raw_volume,
        'pixelspacing':pixelspacing,
        'thickness':thickness
    }

    # -----------------------------
    # Initialize EDV|ESV selection
    # -----------------------------
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}


    # -----------------------------
    # Preprocess / crop if required
    # -----------------------------
    x_min, y_min, x_max, y_max = find_crop_box(np.max(raw_mask[...,[lv_idx, lv_myo_idx]], axis=(-1,-2,-3)), crop_factor=1.5)

    subpixel_resolution = DISPLAY_W//(y_max - y_min)
    subpixel_resolution = min(6, subpixel_resolution)
    st.session_state['subpixel_resolution'] = subpixel_resolution
    
    preprocessed_image = raw_image[y_min:y_max, x_min:x_max, :, :]
    preprocessed_mask = raw_mask[y_min:y_max, x_min:x_max, :, :, :].astype('uint8')
    H, W, D, T, N = preprocessed_mask.shape

    has_masks = np.where(np.sum(preprocessed_mask[...,-1], axis = (0,1,3))>0)[0]
    mid_slice = len(has_masks)//2

    zoom = [st.session_state['subpixel_resolution'],st.session_state['subpixel_resolution'],1,1]

    smoothed_image = cv_zoom(preprocessed_image, zoom = zoom, interpolation=cv2.INTER_LINEAR)

    st.session_state['cache_config_path'] = f"{cache_dir}/config___{sax_series_uid}.json"
    st.session_state['cache_mask_path'] = f"{cache_dir}/masks___{sax_series_uid}.npy"

    if os.path.exists(st.session_state['cache_config_path']) and os.path.exists(st.session_state['cache_mask_path']):
        smoothed_mask = load_cached_mask(st.session_state['cache_mask_path']).astype("uint8")
        cached = True
    else:
        smoothed_mask = cv_zoom_smooth(
            preprocessed_mask,
            zoom=zoom + [1]
        )
        cached = False

    make_video(smoothed_image[:,:,has_masks[mid_slice-3:mid_slice+3],:], smoothed_mask[:,:,has_masks[mid_slice-3:mid_slice+3],:, :] * 0, save_file=edv_esv_gif_path)
    make_video(smoothed_image, smoothed_mask*0, save_file=blank_gif_path)

    gif = Image.open(edv_esv_gif_path)

    st.session_state.preprocessed = {
        "image": preprocessed_image,
        "mask": preprocessed_mask,
        "smooth_image": smoothed_image,
        "smooth_mask": smoothed_mask,
        "H": H,
        "W": W,
        "D": D,
        "T": T,
        "N": N,
        "edv_esv_frames": [frame.copy() for frame in ImageSequence.Iterator(gif)],
        "crop_box": [x_min, y_min, x_max, y_max],
    }

    if cached:
        config = load_config(st.session_state['cache_config_path'])
        confirm_selection(dia_idx=config['dia_idx'], sys_idx=config['sys_idx'])

    # -----------------------------
    # Initialize edited mask
    # -----------------------------
    st.session_state['edited_mask'] = st.session_state.preprocessed["smooth_mask"].copy()
    save_cached_mask(st.session_state['edited_mask'], save_path=st.session_state['cache_mask_path'])
    st.session_state['mask_hash']= mask_hash(st.session_state.preprocessed["mask"])
    st.session_state["brush_mode"] = "Paint ✏️"
    st.session_state["stroke_width"] = "thin"
    st.session_state["edit_made"] = False
    st.session_state['edited_frames'] = None
    st.session_state['cached'] = cached

    st.session_state.initialized_all = True


def cv_zoom(images, zoom, interpolation):
    """
    Resize height and width of a 4D or 5D array using OpenCV. Only H and W are scaled.

    Args
        images (numpy.ndarray): Array of shape (H, W, D, T) or (H, W, D, T, C)
        zoom_factors (list or tuple): Zoom factors for (H, W, D, T, C). Only H and W > 1
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_CUBIC)

    Returns:
        numpy.ndarray: Resized array with height and width scaled, other dimensions unchanged
    """
    h_zoom, w_zoom = zoom[0], zoom[1]

    if images.ndim == 4:
        h, w, d, t = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                resized[..., z, tau] = cv2.resize(images[..., z, tau], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    elif images.ndim == 5:
        h, w, d, t, c = images.shape
        resized = np.zeros((int(h*h_zoom), int(w*w_zoom), d, t, c), dtype=images.dtype)
        for z in range(d):
            for tau in range(t):
                for ch in range(c):
                    resized[..., z, tau, ch] = cv2.resize(images[..., z, tau, ch], (int(w*w_zoom), int(h*h_zoom)), interpolation=interpolation)
    else:
        raise ValueError("Input must be 4D or 5D array.")

    return resized



def cv_zoom_smooth(
    mask,
    zoom,
    sigma=2.0
):
    """
    mask: H,W,D,T,C
    returns: H,W,D,T,C one-hot
    """

    zoomed = cv_zoom(mask.astype(np.float32), zoom, interpolation=cv2.INTER_LINEAR)

    myo  = (zoomed[..., lv_myo_idx] > 0.5).astype(np.float32)
    endo = (zoomed[..., lv_idx] > 0.5).astype(np.float32)

    epi = np.zeros_like(myo, dtype=bool)
    for d in range(myo.shape[2]):
        for t in range(myo.shape[3]):
            epi[..., d, t] = binary_fill_holes(myo[..., d, t].astype(np.uint8))

    epi  = gaussian_filter(epi.astype(np.float32),  sigma=(sigma, sigma, 0, 0)) > 0.5 
    endo = gaussian_filter(endo.astype(np.float32), sigma=(sigma, sigma, 0, 0)) > 0.5

    # Encode labels: 0=bg, 1=endo, 2=myo
    labels = np.zeros(epi.shape, dtype=np.uint8)
    labels[epi]  = lv_myo_idx
    labels[endo] = lv_idx

    # One-hot
    return np.eye(3, dtype=np.uint8)[labels]


def mask_hash(mask_array):
    return hashlib.md5(mask_array.tobytes()).hexdigest()


def load_nii(nii_path):
    file = nib.load(nii_path)
    data = file.get_fdata(caching='unchanged')
    return data

def thicken_close_fill_and_smooth(strokes, stroke_width):
    if strokes is None or not strokes.any():
        return strokes

    # Use power-law scaling for dilation
    dilation_factor = max(1, int(10 / (stroke_width ** 2)))

    # Detect contours to check for nested shapes
    dilated = binary_dilation(strokes, iterations=dilation_factor)
    contours = find_contours(dilated, 0.5)

    has_ring = False
    for i, c1 in enumerate(contours):
        for j, c2 in enumerate(contours):
            if i == j:
                continue
            y1, x1 = c1[:, 0], c1[:, 1]
            y2, x2 = c2[:, 0], c2[:, 1]
            if (y2.min() > y1.min() and y2.max() < y1.max() and
                x2.min() > x1.min() and x2.max() < x1.max()):
                has_ring = True
                break
        if has_ring:
            break

    if has_ring:
        # Dilation + fill + erosion
        closed = binary_dilation(strokes, iterations=dilation_factor)
        filled = binary_fill_holes(closed)
        filled = binary_erosion(filled, iterations=dilation_factor)
        return filled.astype('uint8')
    else:
        return strokes.astype('uint8')




def make_video(image, mask, save_file, mask_frames = 'all', scale=1):
    position = image.shape[2]
    # frame count must be valid for BOTH image and mask (they can differ if the
    # source image/mask had mismatched T); use the common minimum to be safe.
    timesteps = min(image.shape[3], mask.shape[3])

    grid_rows = int(np.sqrt(position) + 0.5)
    grid_cols = (position + grid_rows - 1) // grid_rows

    H, W = image.shape[:2]
    GIF_H = H*GIF_W/W
    H_scaled, W_scaled = round(GIF_H * scale), round(GIF_W * scale)
    img_min, img_max = np.min(image), np.max(image)

    try:
        font = load_font(int(18 * scale))
    except:
        font = ImageFont.load_default()

    frames = []
    if mask_frames == 'all':
        mask_frames = np.arange(timesteps)
    else:
        # drop any explicitly-requested frame that is out of range for image/mask
        mask_frames = [t for t in mask_frames if 0 <= t < timesteps]

    for t in mask_frames:
        canvas = Image.new(
            "RGBA",
            (grid_cols * W_scaled, grid_rows * H_scaled),
            color=(0, 0, 0, 255)
        )

        draw_canvas = ImageDraw.Draw(canvas)

        for idx in range(position):
            row, col = divmod(idx, grid_cols)

            image_slice = ((image[:,:,idx,t] - img_min) / (img_max - img_min + 1e-9) * 255).astype(np.uint8)
            img_rgb = np.stack([image_slice]*3, axis=-1)
            img_pil = Image.fromarray(img_rgb, mode="RGB").convert("RGBA")

            # Resize slice
            img_pil = img_pil.resize((W_scaled, H_scaled), resample=Image.NEAREST)

            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            if t in mask_frames:
                for ch in channels:
                    ch_mask = mask[:,:,idx,t,ch]
                    if np.any(ch_mask):
                        color = np.array(OVERLAY_COLORS[ch], dtype=np.uint8)
                        overlay[ch_mask > 0] = color
            overlay_pil = Image.fromarray(overlay, mode="RGBA").resize((W_scaled, H_scaled), resample=Image.NEAREST)
            img_pil.alpha_composite(overlay_pil)

            draw_tile = ImageDraw.Draw(img_pil)
            draw_tile.rectangle([0,0,int(28*scale), int(22*scale)], fill=(211,211,211,255))
            draw_tile.text((3*scale,2*scale), f"{idx}", fill=(0,0,0,255), font=font)

            canvas.paste(img_pil, (col * W_scaled, row * H_scaled), img_pil)

        draw_canvas.rectangle(
            [canvas.width - int(60*scale), canvas.height - int(20*scale),
             canvas.width, canvas.height],
            fill=(211,211,211,255)
        )
        draw_canvas.text(
            (canvas.width - int(55*scale), canvas.height - int(20*scale)),
            f"{t:02}/{timesteps - 1:02}",
            fill=(0,0,0,255),
            font=font
        )

        frames.append(canvas.convert("RGB"))

    if len(mask_frames) < 5:
        fps = len(mask_frames)/2
    else:
        fps = np.clip(len(mask_frames) / 2, 8, 15)
    imageio.mimsave(save_file, frames, fps=fps, loop=0)


def find_crop_box(mask, crop_factor):
    '''
    Calculated a bounding box that contains the masks inside.

    Parameters:
    mask: np.array
        A binary mask array, which should be the flattened 3D multislice mask, where the pixels in the z-dimension are summed
    crop_factor: float
        A scaling factor for the bounding box
    Returns:
    list
        A list containing the coordinates of the bounding box [x_min, y_min, x_max, y_max]. These co-ordinates can be used to crop each slice of the input multislice image.
    '''
    # Check shape of the input is 2D
    if len(mask.shape) != 2:
        raise ValueError("Input mask must be a 2D array")
    
    y = np.sum(mask, axis=1) # sum the masks across columns of array, returns a 1D array of row totals
    x = np.sum(mask, axis=0) # sum the masks across rows of array, returns a 1D array of column totals

    top = np.min(np.nonzero(y)) - 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. top extent of mask)
    bottom = np.max(np.nonzero(y)) + 1 # Returns the indices of the elements in 1d row totals array that are non-zero, then finds the maximum value and adds 1 (i.e. bottom extent of mask)

    left = np.min(np.nonzero(x)) - 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the minimum value and subtracts 1 (i.e. left extent of mask)
    right = np.max(np.nonzero(x)) + 1 # Returns the indices of the elements in 1d column totals array that are non-zero, then finds the maximum value and adds 1 (i.e. right extent of mask)
    if abs(right - left) > abs(top - bottom):
        largest_side = abs(right - left) # Find the largest side of the bounding box
    else:
        largest_side = abs(top - bottom)
    x_mid = round((left + right) / 2) # Find the mid-point of the x-length of mask
    y_mid = round((top + bottom) / 2) # Find the mid-point of the y-length of mask
    half_largest_side = round(largest_side * crop_factor / 2) # Find half the largest side of the bounding box (crop factor scales the largest side to ensure whole heart and some surrounding is captured)
    x_max, x_min = round(x_mid + half_largest_side), round(x_mid - half_largest_side) # Find the maximum and minimum x-values of the bounding box
    y_max, y_min = round(y_mid + half_largest_side), round(y_mid - half_largest_side) # Find the maximum and minimum y-values of the bounding box
    if x_min < 0:
        x_max -= x_min # if x_min less than zero, expand the x_max value by the absolute value of x_min, to ensure bounding box is same size
        x_min = 0

    if y_min < 0:
        y_max -= y_min # if y_min less than zero, expand the y_max value by the absolute value of y_min, to ensure bounding box is same size
        y_min = 0

    return [x_min, y_min, x_max, y_max]



def calculate_sax_metrics(mask, pixelspacing, thickness, dia_idx, sys_idx):
    voxel_size = pixelspacing ** 2 * thickness / 1000
    volume = np.sum(mask[..., lv_idx], axis=(0,1,2)) * voxel_size
    masses = np.sum(mask[..., lv_myo_idx], axis=(0,1,2)) * voxel_size * 1.05
    mass = masses[dia_idx]
    edv = volume[dia_idx]
    esv = volume[sys_idx]
    sv = edv - esv
    ef = (sv) * 100/edv
    return volume, masses, edv, esv, sv, ef, mass


def _label_vline(ax, x, color, y_pad=0.02):
    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * y_pad
    ax.text(
        x + 0.5,
        y,
        f"{x}",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        rotation=90,
        alpha = 0.75
    )


def plot_volume_mass_curve(
    raw_volume,
    raw_masses,
    edited_volume,
    edited_masses,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.25), sharex=True)

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]
    dia_mass = edited_masses[dia_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"
    mass_color = "#499bed"

    axes[0].plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    axes[0].plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )
    axes[0].set_xticks(np.arange(len(edited_volume)))


    axes[0].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    axes[0].axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[0].axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(axes[0], raw_dia_idx, raw_color)
    _label_vline(axes[0], raw_sys_idx, raw_color)
    _label_vline(axes[0], dia_idx, vol_color)
    _label_vline(axes[0], sys_idx, vol_color)

    axes[0].set_ylabel("Volume (mL)")
    axes[0].set_xlim(0, len(edited_volume) - 1)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    axes[1].plot(frames_raw, raw_masses, color=raw_color, linewidth=2, alpha=0.7)
    axes[1].plot(
        frames_edit,
        edited_masses,
        color=mass_color,
        linewidth=2,
        label=f"Mass: {dia_mass:.1f} g",
    )

    axes[1].axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].axvline(dia_idx, color=mass_color, linestyle="--", linewidth=1.5, alpha=0.75)
    axes[1].set_xticks(np.arange(len(edited_volume)))

    _label_vline(axes[1], raw_dia_idx, raw_color)
    _label_vline(axes[1], dia_idx, mass_color)

    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Mass (g)")
    axes[1].set_xlim(0, len(edited_volume) - 1)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.subplots_adjust(hspace=0.05, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches="tight", dpi = 400)
    plt.close(fig)

def plot_volume_curve(
    raw_volume,
    edited_volume,
    raw_dia_idx,
    raw_sys_idx,
    dia_idx,
    sys_idx,
    save_path,
):

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    frames_raw = np.arange(len(raw_volume))
    frames_edit = np.arange(len(edited_volume))

    edv = edited_volume[dia_idx]
    esv = edited_volume[sys_idx]

    raw_color = "#CBCBCB"
    vol_color = "#f66161"

    ax.plot(frames_raw, raw_volume, color=raw_color, linewidth=2, alpha=0.7)
    ax.plot(
        frames_edit,
        edited_volume,
        color=vol_color,
        linewidth=2,
        label=f"EDV: {edv:.1f} mL | ESV: {esv:.1f} mL",
    )

    ax.axvline(raw_dia_idx, color=raw_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(raw_sys_idx, color=raw_color, linestyle=":", linewidth=1.5, alpha=0.75)
    ax.axvline(dia_idx, color=vol_color, linestyle="--", linewidth=1.5, alpha=0.75)
    ax.axvline(sys_idx, color=vol_color, linestyle=":", linewidth=1.5, alpha=0.75)

    _label_vline(ax, raw_dia_idx, raw_color)
    _label_vline(ax, raw_sys_idx, raw_color)
    _label_vline(ax, dia_idx, vol_color)
    _label_vline(ax, sys_idx, vol_color)

    ax.set_xlabel("Frames")
    ax.set_ylabel("Volume (mL)")
    ax.set_xticks(np.arange(len(edited_volume)))
    ax.set_xlim(0, len(edited_volume) - 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), edgecolor="none")

    plt.savefig(save_path, bbox_inches="tight", dpi=400)
    plt.close(fig)





def wrap(key, min_val, max_val):
    if st.session_state[key] > max_val:
        st.session_state[key] = min_val
    elif st.session_state[key] < min_val:
        st.session_state[key] = max_val

def frame_index_slider(
    T,
    frames,
    initial_idx,
    label,
    disabled_flag,
    key
):
    idx = st.slider(
        f"{label} | *{initial_idx}*",
        -1,
        T,
        value=initial_idx,
        key = key,
        on_change=wrap,
        args=(key, 0, T-1),
        disabled=disabled_flag
    )
    st.image(frames[idx], use_container_width=True)
    return idx



def confirm_selection(dia_idx, sys_idx):
    """Store confirmed EDV|ESV indices in session state."""
    st.session_state.edv_esv_selected.update({
        "dia_idx": dia_idx,
        "sys_idx": sys_idx,
        "confirmed": True
    })

    save_config(st.session_state.edv_esv_selected, st.session_state['cache_config_path'])

    make_video(
        st.session_state.preprocessed["smooth_image"],
        st.session_state.preprocessed["smooth_mask"],
        save_file=full_edited_gif_path,
        mask_frames = [dia_idx, sys_idx]
    )
    gif = Image.open(full_edited_gif_path)
    frames = [f.copy() for f in ImageSequence.Iterator(gif)]
    st.session_state['edited_frames'] = frames



def edv_esv_view():
    """Full EDV|ESV Finder view layout."""
    if "edv_esv_selected" not in st.session_state:
        st.session_state.edv_esv_selected = {"dia_idx": None, "sys_idx": None, "confirmed": False}

    frames= st.session_state.preprocessed['edv_esv_frames']

    if st.session_state.edv_esv_selected['confirmed']:
        display_dia_idx=st.session_state.edv_esv_selected['dia_idx']
        display_sys_idx=st.session_state.edv_esv_selected['sys_idx'] 

    else:
        display_dia_idx=st.session_state.raw['raw_dia_idx']
        display_sys_idx=st.session_state.raw['raw_sys_idx'] 
    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]

    disabled_flag = st.session_state.edv_esv_selected["confirmed"]

    _, col_center,_ = st.columns([0.05,0.9,0.05])
    with col_center:
        col_edv, _, col_esv = st.columns([0.45,0.1,0.45])

        with col_edv:
            dia_idx = frame_index_slider(T, frames, display_dia_idx, 'EDV Index', disabled_flag, key = 'edv')

        with col_esv:
            sys_idx = frame_index_slider(T, frames, display_sys_idx, 'ESV Index', disabled_flag, key = 'esv')

        st.write('')
        if not disabled_flag:
            st.button(
                "Confirm EDV | ESV",
                on_click=lambda: confirm_selection(dia_idx, sys_idx),
                type="primary",
                use_container_width=True
            )
        else:
            st.success("EDV | ESV Confirmed!")



def slice_navigation(D):
    if "slice_idx" not in st.session_state:
        st.session_state.slice_idx = 0
    if "previous_slice_idx" not in st.session_state:
        st.session_state.previous_slice_idx = st.session_state.slice_idx

    # Store previous slice
    previous_d = st.session_state.previous_slice_idx

    # Slider (updates slice_idx immediately)
    st.slider(
        "Slice Index",
        0,
        D - 1,
        key="slice_idx",
    )

    col_prev, col_next = st.columns(2)
    with col_prev:
        st.button(
            "Previous",
            on_click=lambda: st.session_state.update(
                slice_idx=max(0, st.session_state.slice_idx - 1)
            ),
            use_container_width=True,
        )
    with col_next:
        st.button(
            "Next",
            on_click=lambda: st.session_state.update(
                slice_idx=min(D - 1, st.session_state.slice_idx + 1)
            ),
            use_container_width=True,
        )

    # Determine if canvas needs reset
    previous_objects = st.session_state.get('canvas', {}).get('previous_objects', [])
    reset_canvas = previous_d != st.session_state.slice_idx and bool(previous_objects)

    # Update previous slice for next rerun
    st.session_state.previous_slice_idx = st.session_state.slice_idx

    return st.session_state.slice_idx, reset_canvas



def get_overlay(image_slice, mask_state, H, W, N, OVERLAY_COLORS):
    overlay = Image.fromarray(np.stack([image_slice]*3, axis=-1)).convert("RGBA")
    for i in channels:
        ch_mask = mask_state[:, :, i]
        if np.any(ch_mask):
            mask_img = np.zeros((H*st.session_state['subpixel_resolution'], W*st.session_state['subpixel_resolution'], 4), dtype=np.uint8)
            mask_img[ch_mask > 0] = OVERLAY_COLORS[i]
            overlay = Image.alpha_composite(overlay, Image.fromarray(mask_img))
    return overlay


def select_brush(N):
    """Brush selection UI for channel, action, and stroke width."""
    action = st.radio("Brush Stroke Selection", 
                      options=["Paint ✏️", "Erase ✂️"],  
                      index=["Paint ✏️", "Erase ✂️"].index(st.session_state.brush_mode),
                      horizontal=True)
    
    st.session_state['brush_mode'] = action
    stroke_width_map = {"thin":4, "medium":12,"thick":40}

    stroke_width_sel = st.radio("Stroke Width", 
                                options=list(stroke_width_map.keys()),  
                                index= list(stroke_width_map.keys()).index(st.session_state["stroke_width"]), 
                                horizontal=True)
    
    st.session_state['stroke_width'] = stroke_width_sel
    
    if action == "Paint ✏️":
        valid_channels = [i for i in range(N) if i != background_idx]
        channel = st.radio(
            "Mask",
            options=valid_channels,
            format_func=lambda x: BRUSH_LABELS[x],
            index=0,
            horizontal=True
        )

    else:
        channel = 0
    stroke_width = stroke_width_map[stroke_width_sel]
    return channel, action, stroke_width


def normalize(image):
    image = (image - np.min(image))/(np.max(image) - np.min(image))
    return image



# ==================================================================
# 3D renderer for the corrected mask (LV blood pool + myocardium)
# ==================================================================
def _get_threejs_tag():
    """Inline a bundled copy of three.js if one exists (works with no internet),
    otherwise fall back to the CDN."""
    if "_threejs_tag" not in globals():
        local = Path(__file__).parent / "assets" / "three.min.js"
        if local.exists():
            globals()["_threejs_tag"] = "<script>%s</script>" % local.read_text(encoding="utf-8")
        else:
            globals()["_threejs_tag"] = (
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
            )
    return globals()["_threejs_tag"]


_LV_HTML = """
<div id="c" style="width:100%;height:460px;background:#000;"></div>
__THREEJS__
<script>
const meshes = __MESHES__;
const el = document.getElementById('c');
const W = el.clientWidth, H = 460;
let renderer;
try {
    renderer = new THREE.WebGLRenderer({antialias:true});
} catch (e) {
    el.innerHTML = '<p style="color:#fff;padding:20px;font-family:sans-serif">'
                 + '3D rendering unavailable on this display (WebGL not supported).</p>';
    throw e;
}
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(50, W/H, 0.1, 5000);
renderer.setSize(W, H);
el.appendChild(renderer.domElement);
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(1,1,1); scene.add(dl);

// ---- geometry + centre of mass ----
const inner = new THREE.Group();
let sx=0, sy=0, sz=0, vcount=0;
meshes.forEach(m => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(m.vertices, 3));
    g.setIndex(m.faces);
    g.computeVertexNormals();
    const op = (m.opacity === undefined) ? 1.0 : m.opacity;
    const mat = new THREE.MeshPhongMaterial({
        color: m.color, side: THREE.DoubleSide, flatShading: false,
        shininess: 20, transparent: op < 1.0, opacity: op, depthWrite: op >= 1.0,
    });
    inner.add(new THREE.Mesh(g, mat));
    const p = m.vertices;
    for (let i=0;i<p.length;i+=3){ sx+=p[i]; sy+=p[i+1]; sz+=p[i+2]; }
    vcount += p.length/3;
});
const cx=vcount?sx/vcount:0, cy=vcount?sy/vcount:0, cz=vcount?sz/vcount:0;
inner.position.set(-cx, -cy, -cz);

const orient = new THREE.Group(); orient.add(inner); orient.rotation.z = -Math.PI/2;
const pivot  = new THREE.Group(); pivot.add(orient);  pivot.rotation.y  =  Math.PI/2;
scene.add(pivot);

// ---- framing ----
let maxr = 1;
meshes.forEach(m => {
    const p = m.vertices;
    for (let i=0;i<p.length;i+=3){
        const dx=p[i]-cx, dy=p[i+1]-cy, dz=p[i+2]-cz;
        maxr = Math.max(maxr, Math.sqrt(dx*dx+dy*dy+dz*dz));
    }
});
const target = new THREE.Vector3(0,0,0);
cam.position.set(0, 0, maxr*2.5);
cam.lookAt(target);

// ---- mouse: left = rotate, right = pan, wheel = zoom ----
let dragging=false, panning=false, px=0, py=0;
renderer.domElement.addEventListener('mousedown', e=>{
    px=e.clientX; py=e.clientY;
    if (e.button===0) dragging=true; else if (e.button===2) panning=true;
});
window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
window.addEventListener('mousemove', e=>{
    const dx=e.clientX-px, dy=e.clientY-py;
    if (dragging){
        pivot.rotateOnWorldAxis(new THREE.Vector3(0,1,0), dx*0.01);
        pivot.rotateOnWorldAxis(new THREE.Vector3(1,0,0), dy*0.01);
    } else if (panning){
        const dist = cam.position.distanceTo(target);
        const s = dist*0.0015;
        const right = new THREE.Vector3(), up = new THREE.Vector3();
        cam.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        const pan = new THREE.Vector3().addScaledVector(right, -dx*s).addScaledVector(up, dy*s);
        cam.position.add(pan); target.add(pan); cam.lookAt(target);
    }
    px=e.clientX; py=e.clientY;
});
renderer.domElement.addEventListener('wheel', e=>{
    e.preventDefault();
    const dir = new THREE.Vector3().subVectors(cam.position, target);
    dir.multiplyScalar(1 + e.deltaY*0.001);
    cam.position.copy(target).add(dir);
    cam.lookAt(target);
}, {passive:false});

(function animate(){ requestAnimationFrame(animate); renderer.render(scene,cam); })();
</script>
"""


# --- 3D render settings (visualisation only; metrics are unaffected) ---
LV3D_SMOOTH_ITERS = 40      # Taubin passes; volume-preserving
LV3D_DILATE_MM = 11.0        # myocardium ONLY: dilate by this much before interpolating
LV3D_ERODE_MM = 7.0         # myocardium ONLY: erode back by this much afterwards
                            # (equal values -> original wall thickness restored)
LV4D_STEP_SIZE = 3          # coarser mesh for the cine: ~30 frames of payload
LV4D_FPS = 20               # cine playback rate
LV4D_TEMPORAL_SIGMA = 1.0   # smooth the distance field along TIME (frames), cyclic.
                            # Damps frame-to-frame segmentation jitter, but also
                            # damps real contraction (~2pp of EF at 1.0, ~9pp at
                            # 3.0), so keep it low. Visualisation only. 0 = off.
LV_VERT_DP = 2              # round vertices to 0.01mm -> ~40% smaller payload


def _slicewise_sdf(vol_binary, inplane_mm, clamp=10.0):
    """Per-slice 2D signed distance field, in mm, POSITIVE INSIDE the structure.

    Why: linearly interpolating a binary mask between slices only works where the
    shapes overlap. A thin myocardial ring that tapers or drifts between slices
    only partially overlaps its neighbour, and in the non-overlapping part the
    interpolated value peaks at exactly 0.5 -- right on the marching-cubes
    threshold -- so the surface breaks up into holes even though every acquired
    slice contains a complete ring.

    Interpolating a distance field instead makes the shape *morph* between slices,
    which keeps the ring closed. (Shape-based interpolation, Raya & Udupa 1990.)
    """
    from scipy.ndimage import distance_transform_edt

    out = np.empty(vol_binary.shape, dtype=np.float32)
    for d in range(vol_binary.shape[2]):
        b = vol_binary[..., d]
        if not b.any():
            out[..., d] = -clamp                 # wholly outside
        elif b.all():
            out[..., d] = clamp                  # wholly inside
        else:
            d_in = distance_transform_edt(b, sampling=inplane_mm)
            d_out = distance_transform_edt(~b, sampling=inplane_mm)
            out[..., d] = np.clip(d_in - d_out, -clamp, clamp)
    return out


def _taubin_smooth(verts, faces, iterations=10, lam=0.5, mu=-0.53):
    """Taubin surface smoothing.

    Alternates a shrinking Laplacian pass (lam) with an expanding one (mu), so the
    surface is smoothed WITHOUT the volume loss that plain Laplacian smoothing --
    or a big gaussian `sigma` -- would cause. Measured volume drift is <0.5% even
    at 40 iterations, versus -24% for gaussian sigma=2.0 on a thin myocardium.
    """
    import scipy.sparse as sp

    if iterations <= 0 or len(verts) == 0:
        return verts
    n = len(verts)
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])                      # undirected
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0                                     # dedupe repeated edges
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    W = sp.diags(1.0 / deg) @ A                         # neighbour-average operator
    v = verts.astype(np.float64)
    for i in range(iterations):
        f = lam if i % 2 == 0 else mu
        v = v + f * (W @ v - v)
    return v.astype(np.float32)



_LV_4D_HTML = """
<div id="c" style="width:100%;height:430px;background:#000;"></div>
<div style="display:flex;gap:10px;align-items:center;padding:6px 2px;
            font-family:sans-serif;color:#bbb;font-size:12px;">
  <button id="pp" style="width:34px;height:26px;cursor:pointer;background:#333;
          color:#eee;border:1px solid #555;border-radius:4px;">&#9208;</button>
  <input id="sl" type="range" min="0" value="0" style="flex:1;cursor:pointer;">
  <span id="lbl" style="min-width:52px;text-align:right;font-variant-numeric:tabular-nums;"></span>
</div>
__THREEJS__
<script>
const frames = __FRAMES__;
const FPS = __FPS__;

// decode the compact base64 payload (float32 verts, uint16/uint32 faces) into
// typed arrays once, in place, so the rest of the code sees m.vertices / m.faces.
function _b64bytes(s){
    const bin = atob(s); const n = bin.length;
    const buf = new Uint8Array(n);
    for (let i=0;i<n;i++) buf[i]=bin.charCodeAt(i);
    return buf.buffer;
}
frames.forEach(fr => fr.forEach(m => {
    if (m.v !== undefined) {
        m.vertices = new Float32Array(_b64bytes(m.v));
        m.faces = m.f16 ? new Uint16Array(_b64bytes(m.f))
                        : new Uint32Array(_b64bytes(m.f));
        delete m.v; delete m.f;
    }
}));
const el = document.getElementById('c');
const W = el.clientWidth, H = 430;
let renderer;
try {
    renderer = new THREE.WebGLRenderer({antialias:true});
} catch (e) {
    el.innerHTML = '<p style="color:#fff;padding:20px;font-family:sans-serif">'
                 + '3D rendering unavailable on this display (WebGL not supported).</p>';
    throw e;
}
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(50, W/H, 0.1, 5000);
renderer.setSize(W, H);
el.appendChild(renderer.domElement);
renderer.domElement.addEventListener('contextmenu', e => e.preventDefault());
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dl = new THREE.DirectionalLight(0xffffff, 0.6); dl.position.set(1,1,1); scene.add(dl);

// ---- centre of mass over ALL frames, so the heart doesn't jitter during playback ----
let sx=0, sy=0, sz=0, n=0;
frames.forEach(fr => fr.forEach(m => {
    const p = m.vertices;
    for (let i=0;i<p.length;i+=3){ sx+=p[i]; sy+=p[i+1]; sz+=p[i+2]; n++; }
}));
const cx=n?sx/n:0, cy=n?sy/n:0, cz=n?sz/n:0;

// ---- one group per frame, all built up-front; playback just toggles visibility ----
const inner = new THREE.Group();
const groups = frames.map((fr, fi) => {
    const g = new THREE.Group();
    fr.forEach(m => {
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position',
            new THREE.BufferAttribute(m.vertices, 3));
        geo.setIndex(new THREE.BufferAttribute(m.faces, 1));
        geo.computeVertexNormals();
        const op = (m.opacity === undefined) ? 1.0 : m.opacity;
        g.add(new THREE.Mesh(geo, new THREE.MeshPhongMaterial({
            color: m.color, side: THREE.DoubleSide, shininess: 20,
            transparent: op < 1.0, opacity: op, depthWrite: op >= 1.0,
        })));
    });
    g.visible = (fi === 0);
    inner.add(g);
    return g;
});
inner.position.set(-cx, -cy, -cz);
const orient = new THREE.Group(); orient.add(inner); orient.rotation.z = -Math.PI/2;
const pivot  = new THREE.Group(); pivot.add(orient);  pivot.rotation.y  =  Math.PI/2;
scene.add(pivot);

let maxr = 1;
frames.forEach(fr => fr.forEach(m => {
    const p = m.vertices;
    for (let i=0;i<p.length;i+=3){
        const dx=p[i]-cx, dy=p[i+1]-cy, dz=p[i+2]-cz;
        maxr = Math.max(maxr, Math.sqrt(dx*dx+dy*dy+dz*dz));
    }
}));
const target = new THREE.Vector3(0,0,0);
cam.position.set(0, 0, maxr*2.5);
cam.lookAt(target);

// ---- playback (entirely in the browser: no streamlit reruns) ----
const pp = document.getElementById('pp');
const sl = document.getElementById('sl');
const lbl = document.getElementById('lbl');
sl.max = frames.length - 1;
let cur = 0, playing = true;
function show(i){
    groups[cur].visible = false;
    cur = ((i % frames.length) + frames.length) % frames.length;
    groups[cur].visible = true;
    sl.value = cur;
    lbl.textContent = (cur+1) + ' / ' + frames.length;
}
show(0);
setInterval(() => { if (playing) show(cur + 1); }, 1000 / FPS);
pp.onclick = () => { playing = !playing; pp.innerHTML = playing ? '&#9208;' : '&#9654;'; };
sl.oninput = () => { playing = false; pp.innerHTML = '&#9654;'; show(+sl.value); };

// ---- mouse: left = rotate, right = pan, wheel = zoom ----
let dragging=false, panning=false, px=0, py=0;
renderer.domElement.addEventListener('mousedown', e=>{
    px=e.clientX; py=e.clientY;
    if (e.button===0) dragging=true; else if (e.button===2) panning=true;
});
window.addEventListener('mouseup', ()=>{ dragging=false; panning=false; });
window.addEventListener('mousemove', e=>{
    const dx=e.clientX-px, dy=e.clientY-py;
    if (dragging){
        pivot.rotateOnWorldAxis(new THREE.Vector3(0,1,0), dx*0.01);
        pivot.rotateOnWorldAxis(new THREE.Vector3(1,0,0), dy*0.01);
    } else if (panning){
        const s = cam.position.distanceTo(target)*0.0015;
        const right = new THREE.Vector3(), up = new THREE.Vector3();
        cam.matrixWorld.extractBasis(right, up, new THREE.Vector3());
        const pan = new THREE.Vector3().addScaledVector(right, -dx*s).addScaledVector(up, dy*s);
        cam.position.add(pan); target.add(pan); cam.lookAt(target);
    }
    px=e.clientX; py=e.clientY;
});
renderer.domElement.addEventListener('wheel', e=>{
    e.preventDefault();
    const dir = new THREE.Vector3().subVectors(cam.position, target);
    dir.multiplyScalar(1 + e.deltaY*0.001);
    cam.position.copy(target).add(dir);
    cam.lookAt(target);
}, {passive:false});

(function animate(){ requestAnimationFrame(animate); renderer.render(scene,cam); })();
</script>
"""


_LV_STRUCTURES = {
    lv_myo_idx: ("Myocardium", "#00d5d5"),
    lv_idx:     ("Blood Pool", "#ff2b2b"),
}
_SDF_CLAMP = 10.0            # mm; distance field is clipped to +/- this
_MESH_PAD = 1                # pad with "outside" so edge-touching surfaces close


def _iso_plan(spacing, iso_mm):
    """Output spacing + zoom factors for an isotropic render."""
    spacing = np.asarray(spacing, dtype=float)
    if iso_mm is None:
        return spacing, None
    out_spacing = np.array([float(iso_mm)] * 3)
    zoom_factors = spacing / float(iso_mm)
    if np.allclose(zoom_factors, 1.0):
        zoom_factors = None
    return out_spacing, zoom_factors


def _mesh_from_field(field, ch, out_spacing, zoom_factors, step_size,
                     sigma, smooth_iters, dilate_mm, erode_mm):
    """Interpolate a distance field onto the iso grid and mesh its level set."""
    from scipy.ndimage import zoom

    if zoom_factors is not None:
        field = zoom(field, zoom_factors, order=1)
    field = np.pad(field, _MESH_PAD, mode='constant', constant_values=-_SDF_CLAMP)
    if sigma:
        field = gaussian_filter(field, sigma=sigma)

    # dilate/erode the myocardium only; the blood pool is a filled blob that
    # interpolates cleanly, so it is meshed exactly as segmented.
    level = -(float(dilate_mm) - float(erode_mm)) if ch == lv_myo_idx else 0.0

    try:
        verts, faces, _, _ = marching_cubes(
            field, level=level, spacing=tuple(out_spacing), step_size=step_size)
    except (ValueError, RuntimeError):
        return None                      # no level crossing -> nothing to draw
    verts = verts - _MESH_PAD * out_spacing
    verts = _taubin_smooth(verts, faces, iterations=smooth_iters)
    name, color = _LV_STRUCTURES[ch]
    return {
        "name": name, "color": color,
        # rounding to 0.01mm is far finer than the data and roughly halves the
        # JSON payload (a raw float dumps ~17 significant figures)
        "vertices": np.round(verts, LV_VERT_DP).astype(np.float32).ravel().tolist(),
        "faces": faces.astype(np.uint32).ravel().tolist(),
    }


def build_lv_geometry(mask_onehot, idx, spacing=(1.0, 1.0, 1.0), step_size=2,
                      sigma=0.0, iso_mm=None, smooth_iters=30,
                      dilate_mm=0.0, erode_mm=0.0):
    """Marching-cubes surfaces for one timepoint of the corrected mask.

    mask_onehot : (H, W, D, T, N) one-hot edited mask
    idx         : timepoint to render (dia_idx or sys_idx)
    spacing     : (row_mm, col_mm, slice_mm) of the mask grid
    iso_mm      : if given, resample the mask onto an isotropic `iso_mm` grid
                  before meshing. SAX data is ~20-30:1 anisotropic (thin pixels,
                  thick slices), which makes marching cubes produce a stack of
                  flat slabs. Interpolating the slice axis up to the in-plane
                  resolution removes that and gives a smoothly tapered wall.
                  Pass None to mesh the mask exactly as stored.
    sigma       : optional blur of the distance field. Defaults to OFF: blurring
                  erodes thin structures from both sides and punches holes in the
                  myocardial ring. Use `smooth_iters` to smooth instead.
    smooth_iters: Taubin surface-smoothing passes applied after meshing. This is
                  the safe smoothing knob -- raise it freely (10-40), it barely
                  touches the enclosed volume.
    dilate_mm   : MYOCARDIUM ONLY -- dilate by this many mm before interpolating,
                  so a thin ring overlaps itself between slices.
    erode_mm    : MYOCARDIUM ONLY -- erode back by this many mm afterwards, to
                  restore the original wall thickness.

                  Both are applied by shifting the marching-cubes level, which is
                  exact: thresholding a euclidean distance field at -k IS k-mm
                  dilation by definition, and linear interpolation is affine
                  (zoom(sdf + k) == zoom(sdf) + k), so it makes no difference
                  whether the shift happens before or after interpolation.
                  Net level = -(dilate_mm - erode_mm).

                  NOTE: because both are shifts of the same field, dilate_mm ==
                  erode_mm cancels exactly and returns the un-dilated surface.
                  That is not a bug -- interpolating a distance field already
                  behaves as if the shape were dilated by every amount at once,
                  which is why the ring stays closed without the round-trip.
                  Set erode_mm < dilate_mm to leave the wall visibly thicker.

                  Visualisation only: the mask the metrics are computed from is
                  never touched.
    """
    out_spacing, zoom_factors = _iso_plan(spacing, iso_mm)
    spacing = np.asarray(spacing, dtype=float)

    meshes = []
    for ch in _LV_STRUCTURES:
        vol = mask_onehot[:, :, :, idx, ch] > 0.5
        if not vol.any():
            continue
        # Shape-based interpolation: build a distance field (positive inside) and
        # interpolate THAT between slices, so thin rings morph instead of breaking.
        field = _slicewise_sdf(vol, spacing[:2], clamp=_SDF_CLAMP)
        m = _mesh_from_field(field, ch, out_spacing, zoom_factors, step_size,
                             sigma, smooth_iters, dilate_mm, erode_mm)
        if m is not None:
            meshes.append(m)
    return meshes


def build_lv_geometry_4d(mask_onehot, spacing=(1.0, 1.0, 1.0), iso_mm=None,
                         step_size=None, smooth_iters=30,
                         dilate_mm=0.0, erode_mm=0.0,
                         temporal_sigma=None, progress=None):
    """Surfaces for EVERY timepoint -> one entry per cine frame.

    Intended to be run on the ORIGINAL segmentation, not the edited mask.

    temporal_sigma : gaussian smoothing of the distance field ALONG TIME, in
        frames, applied cyclically (mode='wrap') because a cardiac cine loops.
        Smoothing the distance field rather than the binary mask matters for the
        same reason it does spatially: averaging 0/1 across frames strands a
        thin, moving ring at exactly 0.5 and punches holes in it, whereas
        distances blend into a clean intermediate surface.

        This damps genuine contraction as well as jitter (~2pp of EF at sigma=1,
        ~9pp at sigma=3), so keep it low. Visualisation only. None -> the
        LV4D_TEMPORAL_SIGMA default.
    """
    from scipy.ndimage import gaussian_filter1d

    if step_size is None:
        step_size = LV4D_STEP_SIZE
    if temporal_sigma is None:
        temporal_sigma = LV4D_TEMPORAL_SIGMA

    spacing = np.asarray(spacing, dtype=float)
    out_spacing, zoom_factors = _iso_plan(spacing, iso_mm)
    T = mask_onehot.shape[3]
    frames = [[] for _ in range(T)]

    total = len(_LV_STRUCTURES) * T
    done = 0
    for ch in _LV_STRUCTURES:
        # distance field for every frame, stacked on a time axis -> (H, W, D, T)
        fields = np.stack(
            [_slicewise_sdf(mask_onehot[:, :, :, t, ch] > 0.5, spacing[:2],
                            clamp=_SDF_CLAMP) for t in range(T)],
            axis=-1,
        )
        if temporal_sigma:
            # cyclic: in a cine loop, frame T-1 is adjacent to frame 0
            fields = gaussian_filter1d(fields, sigma=float(temporal_sigma),
                                       axis=-1, mode='wrap')
        for t in range(T):
            m = _mesh_from_field(fields[..., t], ch, out_spacing, zoom_factors,
                                 step_size, 0.0, smooth_iters, dilate_mm, erode_mm)
            if m is not None:
                frames[t].append(m)
            done += 1
            if progress is not None:
                progress(done / total)
        del fields
    return frames


def _render_lv_3d(edited_mask, idx, idx_label):
    """3D view of the corrected mask for the currently selected frame."""
    import streamlit.components.v1 as components

    sub = st.session_state.get('subpixel_resolution', 1)
    ps = float(st.session_state.raw['pixelspacing'])
    th = float(st.session_state.raw['thickness'])
    spacing = (ps / sub, ps / sub, th)          # in-plane was upsampled by `sub`
    # Render on an isotropic grid at the TRUE in-plane resolution (`ps`). The
    # `/sub` upsampling only exists to make the drawing canvas big, so meshing at
    # `ps` costs ~200x less than meshing at `ps/sub` and loses no real detail.
    iso_mm = ps

    c_upd, c_myo, c_bp = st.columns([1.2, 1, 1])
    with c_upd:
        update = st.button("🔄 Update 3D", use_container_width=True, key="lv3d_update")
    with c_myo:
        myo_solid = st.checkbox("Myocardium", value=False, key="lv3d_myo",
                                help="Solid when ticked, translucent when not")
    with c_bp:
        bp_solid = st.checkbox("Blood Pool", value=True, key="lv3d_bp")

    need_render = (
        update
        or st.session_state.get('lv_geom') is None
        or st.session_state.get('lv_geom_idx') != idx
    )
    if need_render:
        with st.spinner(f"Building 3D surfaces ({idx_label})..."):
            st.session_state['lv_geom'] = build_lv_geometry(
                edited_mask, idx, spacing, iso_mm=iso_mm,
                smooth_iters=LV3D_SMOOTH_ITERS,
                dilate_mm=LV3D_DILATE_MM, erode_mm=LV3D_ERODE_MM)
            st.session_state['lv_geom_idx'] = idx
            st.session_state['lv_geom_stale'] = False

    meshes = st.session_state.get('lv_geom') or []
    if not meshes:
        st.info(f"No mask to render for {idx_label}.")
        return

    opacity = {"Myocardium": 1.0 if myo_solid else 0.18,
               "Blood Pool": 1.0 if bp_solid else 0.18}
    meshes_alpha = [{**m, "opacity": opacity.get(m["name"], 1.0)} for m in meshes]

    html = _LV_HTML.replace("__MESHES__", json.dumps(meshes_alpha))
    html = html.replace("__THREEJS__", _get_threejs_tag())
    components.html(html, height=480, scrolling=False)

    st.caption(f"Showing **{idx_label}** — drag to rotate, right-drag to pan, scroll to zoom")
    if st.session_state.get('lv_geom_stale'):
        st.caption("⚠️ Edits since last render — click Update 3D")



def _render_lv_4d():
    """Animated 3D view of the ORIGINAL 4D segmentation, across all cine frames.

    Deliberately reads st.session_state.preprocessed['mask'] -- the segmentation as
    it came out of the model. It ignores every correction made in the editor, so it
    never needs rebuilding and is a stable reference to compare edits against.
    """
    import streamlit.components.v1 as components

    ps = float(st.session_state.raw['pixelspacing'])
    th = float(st.session_state.raw['thickness'])
    # NB: preprocessed['mask'] is at native in-plane resolution (NOT upsampled by
    # `subpixel_resolution`), so spacing is (ps, ps, th) and iso_mm=ps means the
    # zoom factors are (1, 1, th/ps) -- i.e. the SLICE axis only is interpolated.
    spacing = (ps, ps, th)

    orig_mask = st.session_state.preprocessed.get("mask")
    if orig_mask is None:
        st.info("Original segmentation not available for this case.")
        return
    T = orig_mask.shape[3]

    c_myo, c_bp = st.columns([ 1, 1])
    with c_myo:
        myo_solid = st.checkbox("Myocardium", value=False, key="lv4d_myo",
                                help="Solid when ticked, translucent when not")
    with c_bp:
        bp_solid = st.checkbox("Blood Pool", value=True, key="lv4d_bp")

    # the original mask never changes, so this is built once per case
    uid = st.session_state.get('sax_series_uid')
    if st.session_state.get('lv_geom_4d_uid') != uid:
        bar = st.progress(0.0, text=f"Building {T} cine frames...")
        try:
            st.session_state['lv_geom_4d'] = build_lv_geometry_4d(
                orig_mask, spacing, iso_mm=ps,
                smooth_iters=LV3D_SMOOTH_ITERS,
                dilate_mm=LV3D_DILATE_MM, erode_mm=LV3D_ERODE_MM,
                progress=lambda f: bar.progress(f, text=f"Building cine frames... {int(f*T)}/{T}"),
            )
            st.session_state['lv_geom_4d_uid'] = uid
        finally:
            bar.empty()

    frames = st.session_state.get('lv_geom_4d') or []
    frames = [f for f in frames if f]
    if not frames:
        st.info("Nothing to render in the original segmentation.")
        return

    opacity = {"Myocardium": 1.0 if myo_solid else 0.18,
               "Blood Pool": 1.0 if bp_solid else 0.18}

    # Compact payload: vertices as base64 float32, faces as base64 uint16 (when the
    # mesh has < 65536 verts) or uint32. JSON number-lists cost ~5-7 chars per
    # value; binary is 2-4 bytes, cutting the payload ~3x so the 4D cine stays
    # under Streamlit's message-size limit. Faces CANNOT be shared across frames:
    # marching cubes yields different topology as the ventricle deforms.
    frames_alpha = []
    for fr in frames:
        packed = []
        for m in fr:
            verts = np.asarray(m["vertices"], dtype=np.float32)
            faces = np.asarray(m["faces"], dtype=np.uint32)
            nverts = verts.size // 3
            small = nverts < 65536
            fdtype = np.uint16 if small else np.uint32
            packed.append({
                "name": m["name"],
                "color": m["color"],
                "opacity": opacity.get(m["name"], 1.0),
                "v": base64.b64encode(verts.tobytes()).decode("ascii"),
                "f": base64.b64encode(faces.astype(fdtype).tobytes()).decode("ascii"),
                "f16": bool(small),
            })
        frames_alpha.append(packed)

    html = _LV_4D_HTML.replace("__FRAMES__", json.dumps(frames_alpha))
    html = html.replace("__FPS__", str(LV4D_FPS))
    html = html.replace("__THREEJS__", _get_threejs_tag())
    components.html(html, height=500, scrolling=False)

    st.caption(f"Original segmentation, all {len(frames)} frames — **edits are not shown here**")


def mask_editor_view():
    """Efficient Mask Editor with controlled reruns and canvas caching."""
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    H, W, D, T, N = [st.session_state.preprocessed[k] for k in ["H","W","D","T","N"]]
    image = st.session_state.preprocessed["smooth_image"]
    edited_mask = st.session_state['edited_mask']
    dia_idx = st.session_state.edv_esv_selected["dia_idx"]
    sys_idx = st.session_state.edv_esv_selected["sys_idx"]

    col1, col2, col3 = st.columns([1,1.5,1.5])

    with col1:
        channel, action, stroke_width = select_brush(N)
        st.divider()
        idx_label = st.radio("Frame", ["End-Diastole","End-Systole"], index=0, horizontal=True)
        d, reset_canvas = slice_navigation(D)

    idx = dia_idx if idx_label == "End-Diastole" else sys_idx

    # Normalize slice once per display
    img_slice = image[:, :, d, idx]
    image_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
    mask_slice = edited_mask[:, :, d, idx, :]

    with col2:
        edit_mode = st.radio('Segmentation Editor', ['Editor','Viewer'], index = 0, horizontal=True)
        stroke_color = f"rgba{OVERLAY_COLORS[background_idx][:3]+(0.8,)}" if action == "Erase ✂️" else f"rgba{OVERLAY_COLORS[channel][:3]+(0.65,)}"

        if edit_mode == 'Viewer':
            st.image(image_slice, width=DISPLAY_W)
        else:
            # Initialize canvas state
            if 'canvas' not in st.session_state:
                st.session_state['canvas'] = {
                    'previous_d': d,
                    'previous_objects': [],
                    'previous_sig': None,
                    'clear_gen': 0,
                }

            # STABLE key: does NOT include d, so the canvas is never unmounted and
            # remounted when the slice/frame changes -- that remount was what made
            # the canvas go blank. Strokes are cleared via initial_drawing instead.
            canvas_key = "editor_canvas"

            # The canvas component wipes itself whenever `initial_drawing` differs
            # from the value it last received. So it must be passed on EVERY render
            # and stay byte-identical between clears -- otherwise the rerun that
            # follows the user's first stroke changes it and silently erases that
            # stroke. `clear_gen` is bumped ONLY when the slice/frame changes, which
            # is exactly when we want the strokes cleared.
            sig = (d, idx)
            if st.session_state['canvas'].get('previous_sig') != sig:
                st.session_state['canvas']['previous_sig'] = sig
                st.session_state['canvas']['clear_gen'] += 1

            canvas_result = st_canvas(
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=get_overlay(image_slice, mask_slice, H, W, N, OVERLAY_COLORS),
                update_streamlit=True,
                height = H*DISPLAY_W/W,
                width = DISPLAY_W,
                drawing_mode='freedraw',
                key=canvas_key,
                initial_drawing={
                    "version": "4.4.0",
                    "objects": [],
                    "clear_gen": st.session_state['canvas']['clear_gen'],
                },
            )

            # Track current objects
            current_objects = []
            if canvas_result and canvas_result.json_data:
                current_objects = canvas_result.json_data.get("objects", [])
            st.session_state['canvas']['previous_objects'] = current_objects
            st.session_state['canvas']['previous_d'] = d

            # Save / clear buttons (trigger rerun only here)
            col_save, col_clear = st.columns([1, 0.15])
            with col_save:
                save_contour = st.button('Save Contour', type='primary', use_container_width=True)
                if save_contour and canvas_result and canvas_result.image_data is not None and current_objects:
                    brush_data = np.array(canvas_result.image_data)
                    rgb = brush_data[:, :, :3].astype(np.float32)
                    alpha = brush_data[:, :, 3].astype(np.float32) / 255.0

                    overlay_colors_list = np.array([color[:3] for color in OVERLAY_COLORS.values()], dtype=np.float32)
                    overlay_channels = list(OVERLAY_COLORS.keys())

                    h, w, _ = rgb.shape
                    rgb_flat = rgb.reshape(-1, 3)
                    alpha_flat = alpha.flatten()
                    distances = np.linalg.norm(rgb_flat[:, None, :] - overlay_colors_list[None, :, :], axis=-1)
                    closest_idx = np.argmin(distances, axis=1)

                    mask_flat = np.zeros((h*w, len(overlay_channels)), dtype=np.uint8)
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_flat[:, idx_color] = ((closest_idx == idx_color) & (alpha_flat > 0)).astype(np.uint8)

                    masks = []
                    for idx_color, ch in enumerate(overlay_channels):
                        mask_bool = mask_flat[:, idx_color].reshape(h, w)
                        mask_bool = thicken_close_fill_and_smooth(mask_bool, stroke_width)
                        masks.append(mask_bool)

                    combined_mask = np.stack(masks, axis=-1)
                    for idx_color, ch in enumerate(overlay_channels):
                        resized_mask = np.array(
                            Image.fromarray(combined_mask[:, :, idx_color]).resize(
                                (W*st.session_state['subpixel_resolution'], H*st.session_state['subpixel_resolution']),
                                resample=Image.NEAREST
                            )
                        )
                        edited_mask[:, :, d, idx, :][resized_mask > 0] = 0
                        edited_mask[:, :, d, idx, ch][resized_mask > 0] = 1

                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()



            with col_clear:
                if st.button('❌', use_container_width=True):
                    edited_mask[:, :, d, idx, :] = 0
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.session_state['edit_made'] = True
                    st.rerun()
            
            st.divider()
            st.caption('Dilation and Erosion')
            col_expand, col_shrink, _, col_expand_myo, col_shrink_myo = st.columns([1,1,1,1,1])

            with col_expand:
                if st.button("🔴" + " :material/north:", use_container_width=True, key="dilate_bp"):
                    # Step 1: dilate last channel
                    lv_channel = st.session_state['edited_mask'][:, :, d, idx, lv_idx]
                    dilated_last = binary_dilation(lv_channel)
                    edited_mask[:, :, d, idx, lv_myo_idx] = edited_mask[:, :, d, idx, lv_myo_idx] & (~dilated_last)
                    # Step 3: assign updated last channel
                    edited_mask[:, :, d, idx, -1] = dilated_last
                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()
                                
            with col_shrink:
                if st.button("🔴" + " :material/south:", use_container_width=True, key="erode_bp"):
                    lv_channel = st.session_state['edited_mask'][:, :, d, idx, lv_idx]
                    eroded_lv = binary_erosion(lv_channel, iterations=1)
                    
                    # ring = original LV minus eroded LV
                    lv_ring = lv_channel & (~eroded_lv)
                    
                    # add only the ring to myocardium
                    edited_mask[:, :, d, idx, lv_myo_idx] = (
                        edited_mask[:, :, d, idx, lv_myo_idx] | lv_ring
                    )
                    
                    # UPDATE: assign the eroded LV back to the LV channel
                    edited_mask[:, :, d, idx, lv_idx] = eroded_lv
                    
                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()

            with col_expand_myo:
                if st.button("🔵" + " :material/north:", use_container_width=True, key="dilate_myo"):
                    # Get epicardium = LV blood pool + myocardium
                    lv_channel = st.session_state['edited_mask'][:, :, d, idx, lv_idx]
                    myo_channel = st.session_state['edited_mask'][:, :, d, idx, lv_myo_idx]
                    epicardium = lv_channel | myo_channel
                    
                    # Dilate epicardium outward
                    dilated_epi = binary_dilation(epicardium, iterations=1)
                    
                    # New outer ring = dilated epicardium minus original epicardium
                    outer_ring = dilated_epi & (~epicardium)
                    
                    # Add outer ring to myocardium
                    edited_mask[:, :, d, idx, lv_myo_idx] = myo_channel | outer_ring
                    
                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()

            with col_shrink_myo:
                if st.button("🔵" + " :material/south:", use_container_width=True, key="erode_myo"):
                    # Get epicardium = LV blood pool + myocardium
                    lv_channel = st.session_state['edited_mask'][:, :, d, idx, lv_idx]
                    myo_channel = st.session_state['edited_mask'][:, :, d, idx, lv_myo_idx]
                    epicardium = lv_channel | myo_channel
                    
                    # Erode epicardium inward
                    eroded_epi = binary_erosion(epicardium, iterations=1)
                    
                    # New myocardium = eroded epicardium minus LV blood pool
                    edited_mask[:, :, d, idx, lv_myo_idx] = eroded_epi & (~lv_channel)
                    
                    st.session_state['edit_made'] = True
                    save_cached_mask(edited_mask, save_path=st.session_state['cache_mask_path'])
                    st.rerun()


    # ---------- right column preview ----------
    with col3:
        _vm_opts = ["Static", "3D", "4D", "4CH"]
        # Persist the choice in a SEPARATE key (not the widget key): an immediate
        # st.rerun() elsewhere (e.g. Save Contour) discards a widget's just-set
        # state. The on_change callback writes the persistent key AT click time,
        # so switching is single-click AND survives forced reruns.
        if "view_mode_persist" not in st.session_state:
            st.session_state["view_mode_persist"] = "Static"

        def _sync_view_mode():
            st.session_state["view_mode_persist"] = st.session_state["view_mode_radio"]

        view_mode = st.radio(
            "Corrected Mask",
            _vm_opts,
            index=_vm_opts.index(st.session_state["view_mode_persist"]),
            horizontal=True,
            key="view_mode_radio",
            on_change=_sync_view_mode,
        )
        view_mode = st.session_state["view_mode_persist"]

        if st.session_state.get("edited_frames") is None or st.session_state["edit_made"]:
            make_video(
                image,
                edited_mask,
                save_file=full_edited_gif_path,
                mask_frames=[dia_idx, sys_idx],
            )
            gif = Image.open(full_edited_gif_path)
            st.session_state["edited_frames"] = [f.copy() for f in ImageSequence.Iterator(gif)]
            st.session_state["edit_made"] = False
            # the mask just changed, so any cached 3D surface is now out of date
            st.session_state["lv_geom_stale"] = True

        if view_mode == "Static":
            view_image = st.session_state["edited_frames"][0 if idx_label == "End-Diastole" else 1]
            st.image(view_image, width=int(DISPLAY_W * 1.5))
        elif view_mode == "3D":
            # 3D surfaces of the corrected mask for the frame selected above
            _render_lv_3d(edited_mask, idx, idx_label)
        elif view_mode == "4D":
            # 4D cine of the ORIGINAL segmentation (ignores corrections)
            _render_lv_4d()
        else:
            # 4CH cross-reference: where the current SAX slice sits on a 4CH view,
            # with the 4CH background synced to the SAX cardiac phase (idx of T)
            import align_4ch
            align_4ch.render_4ch_view(d, sax_frame=idx, sax_nframes=T)

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(review_list_path):
                download_review_csv(review_list_path)
            
        with col2: 
            read_or_create_review_csv(review_list_path, patient = st.session_state['patient'], study_date = st.session_state['study_date'], sax_series_uid = st.session_state['sax_series_uid'])
            if os.path.exists(review_list_path):
                df = pd.read_csv(review_list_path)
                if st.session_state['sax_series_uid'] in df["sax_series_uid"].values:
                    st.warning(f"{st.session_state['patient']} | {st.session_state['study_date']} in Review")