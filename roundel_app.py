# --------------------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------------------
from roundel_utils import *
import sax_prep
st.set_page_config(page_title="Roundel", page_icon="⭕️", layout='wide')

# -----------------------------
# Single hardcoded case (no dropdown). Build image___/masks___/saxdf___ from the
# SAX image + mask .nii.gz (paths set in sax_prep.py), then load that one case.
# -----------------------------
data_path = 'data/case'
sax_series_uid = sax_prep.ensure_case_data(data_path)
display_uid = sax_prep.get_display_uid()          # real series UID for display

st.write('# Roundel App (2D)')

patient, study_date, description = display_uid, '01-01-2020', 'short axis cine stack'

initialize_app(data_path, patient, study_date, sax_series_uid, preprocess=True)
pixelspacing, thickness = st.session_state.raw['pixelspacing'], st.session_state.raw['thickness']

# Display metadata in the app
st.markdown(f"**SAX Series UID:** {display_uid} | **Patient:** {patient} | **Study Date:** {study_date}")
st.markdown(f"**Description:** {description} | **Pixel Size**: {pixelspacing} x {pixelspacing}mm | **Slice Thickness**: {thickness} mm")

# --------------------------------------------------------------
# App
# --------------------------------------------------------------

view = st.segmented_control(
    "Tab",
    options=["EDV/ESV Finder 🔍", "Mask Editor 📝", "Final Result ✅"],
    default = "EDV/ESV Finder 🔍",
    label_visibility='hidden'
)
st.divider()

# --------------------------------------------------------------
# EDV/ESV Finder 
# --------------------------------------------------------------
if view == "EDV/ESV Finder 🔍":
    edv_esv_view()


# --------------------------------------------------------------
# Mask Editor 
# --------------------------------------------------------------


if view == "Mask Editor 📝":
    # try:
    mask_editor_view()
    # except:
    #     st.rerun()

# --------------------------------------------------------------
# Final Result
# --------------------------------------------------------------

if view == "Final Result ✅":
    raw = st.session_state.raw
    preprocessed = st.session_state.preprocessed

    raw_image = raw["image"]
    raw_mask = raw["mask"]
    raw_edv = raw["raw_edv"]
    raw_esv = raw["raw_esv"]
    raw_mass = raw["raw_mass"]
    raw_ef = raw["raw_ef"]
    preprocessed_image = preprocessed["image"]
    H, W, D, T, N = [preprocessed[k] for k in ["H","W","D","T","N"]]

    edited_mask=st.session_state['edited_mask']
    x_min, y_min, x_max, y_max = preprocessed['crop_box']
    dia_idx=st.session_state.edv_esv_selected['dia_idx']
    sys_idx=st.session_state.edv_esv_selected['sys_idx']
    
    if not st.session_state.edv_esv_selected["confirmed"]:
        st.error("Select and confirm EDV/ESV first.")
        st.stop()

    edited_mask = cv_zoom(edited_mask, 
                          zoom = [1/st.session_state['subpixel_resolution'],1/st.session_state['subpixel_resolution'],1,1], 
                          interpolation=cv2.INTER_NEAREST)

    final_gif_path = f'results/gifs/{sax_series_uid}.gif'
    # Compute metrics
    volume, masses, edv, esv, sv, ef, mass = calculate_sax_metrics(
        edited_mask, pixelspacing, thickness, dia_idx, sys_idx)

    # Create full-size arrays
    final_mask_2d = np.zeros(raw_mask.shape, dtype=raw_mask.dtype)
    final_mask_2d[y_min:y_max, x_min:x_max, :, [dia_idx, sys_idx], 1:] = edited_mask[:, :, :, [dia_idx, sys_idx], 1:]
    final_mask_2d = np.argmax(final_mask_2d, -1)
    
    make_video(
        preprocessed_image,
        edited_mask,
        mask_frames=[dia_idx, sys_idx],
        save_file=edited_gif_path
    )

    make_video(
        raw_image,
        np.eye(N, dtype=np.uint8)[final_mask_2d],
        save_file=final_gif_path, 
        mask_frames=[dia_idx, sys_idx],
        scale = 1.5
    )

    col1, _, col2, col3 = st.columns([0.08, 0.05,0.2, 0.3])
    with col1:
        st.caption('Metrics')
        st.metric("End-Diastolic Volume", f"{edv:.1f} mL",
                  delta=None if edv==raw_edv else f"{edv-raw_edv:.1f} mL")
        st.metric("End-Systolic Volume", f"{esv:.1f} mL",
                  delta=None if esv==raw_esv else f"{esv-raw_esv:.1f} mL")
        st.metric("Ejection Fraction", f"{ef:.1f} %",
                  delta=None if round(ef,1)==round(raw_ef,1) else f"{ef-raw_ef:.1f} %")
        st.metric("Myocardial Mass", f"{mass:.1f} g",
                  delta=None if mass==raw_mass else f"{mass-raw_mass:.1f} g")


        if st.button('Save Masks and Metrics', type='primary', use_container_width=True):
            st.success('Masks and Metrics Saved!')
            save_mask(final_mask_2d, f'results/masks/{sax_series_uid}.nii.gz')
            
            if os.path.exists(st.session_state['cache_mask_path']):
                os.remove(st.session_state['cache_mask_path'])

            df = pd.DataFrame({
                "sax_series_uid": [sax_series_uid],
                "edv_frame": [dia_idx],
                "esv_frame": [sys_idx],

                "edv": [edv],
                "esv": [esv],
                "stroke_volume": [sv],
                "ejection_fraction": [ef],
                "mass": [mass],

                "pixelspacing": [pixelspacing],
                "thickness": [thickness],

                "num_slices": [edited_mask.shape[2]],
                "num_frames": [edited_mask.shape[3]],
                }).to_csv(f'results/edited_sax_df/{sax_series_uid}.csv', index = False)

    with col2:
        st.caption('Final Cropped Mask')
        st.image(edited_gif_path)
    

    with col3:
        st.caption('Final Full-Sized Mask')
        st.image(final_gif_path)