import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace

# --- Initialize session state ---
if "face_data" not in st.session_state:
    st.session_state.face_data = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "files" not in st.session_state:
    st.session_state.files = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "cluster_labels" not in st.session_state:
    st.session_state.cluster_labels = []

st.set_page_config(page_title="Face Count App", layout="wide")
st.title("🎭 Face Recognition & Photo Count")
st.write("Upload images. Faces will be grouped using deep learning; assign custom names with preview.")

# Load face detector (Haar cascade, for initial face crop)
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

face_detector = load_face_detector()

def extract_face_embedding(face_img_rgb):
    """Given cropped face in RGB np.array, extract DeepFace embedding."""
    try:
        result = DeepFace.represent(face_img_rgb, model_name='Facenet', enforce_detection=False)
        return np.array(result[0]['embedding'])
    except Exception as e:
        return None

uploaded = st.file_uploader(
    "📁 Upload images (jpg/png), multiple files or drag-and-drop folder",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded:
    new_files = [f for f in uploaded if f not in st.session_state.files]
    if new_files:
        total_bytes = sum(f.size for f in st.session_state.files) + sum(f.size for f in new_files)
        limit = 1_000_000_000
        if total_bytes <= limit:
            st.session_state.files.extend(new_files)
            st.session_state.processed = False
            st.success(f"✅ Added {len(new_files)} new images. Total: {len(st.session_state.files)} images")
        else:
            st.error("❌ Upload exceeds 1 GB limit. Remove some files.")

if st.session_state.files:
    st.info(f"📊 Currently loaded: {len(st.session_state.files)} images")

# --- Sensitivity slider ---
sensitivity = st.slider(
    "🎚️ Face grouping sensitivity (lower = group more faces together)",
    min_value=1, 
    max_value=10, 
    value=3, 
    step=1,
    help="Lower values group similar faces together. Higher values create more separate groups."
)

if st.button("🚀 Process Images", type="primary") and st.session_state.files:
    st.session_state.face_data = []
    st.session_state.counts = {}
    st.session_state.cluster_labels = []

    progress = st.progress(0)
    status = st.empty()
    n = len(st.session_state.files)

    all_features = []
    all_face_images = []

    for i, file in enumerate(st.session_state.files):
        status.text(f"Processing image {i+1}/{n}: {file.name}")

        img_bytes = file.read()
        file.seek(0)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(80, 80),
            maxSize=(500, 500),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            # Crop and convert for DeepFace
            face_roi_color = image[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
            embedding = extract_face_embedding(face_rgb)
            if embedding is None:
                continue
            all_features.append(embedding)
            face_pil = Image.fromarray(face_rgb)
            all_face_images.append(face_pil)
        progress.progress((i+1)/n)

    status.text("🔍 Grouping similar faces...")

    if all_features:
        X = np.array(all_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        distance_threshold = sensitivity * 0.7  # Lower scale for neural embedding
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'
        ).fit(X_scaled)

        st.session_state.cluster_labels = clustering.labels_
        for idx, (face_img, label) in enumerate(zip(all_face_images, clustering.labels_)):
            st.session_state.face_data.append({
                'image': face_img,
                'cluster': label,
                'index': idx
            })
        unique_labels = sorted(set(clustering.labels_))
        for label in unique_labels:
            person_name = f"Person {label + 1}"
            count = list(clustering.labels_).count(label)
            st.session_state.counts[person_name] = count
        st.session_state.labels = list(st.session_state.counts.keys())
        st.session_state.processed = True
        status.empty()
        st.success(f"✅ Processing complete! Found {len(all_features)} faces in {len(unique_labels)} groups.")
    else:
        status.empty()
        st.warning("⚠️ No faces detected. Try different images or lower the sensitivity.")

# --- Name editing with preview --- (no changes needed)
if st.session_state.processed and st.session_state.labels:
    st.subheader("✏️ Edit Person Names")
    st.write("Preview the faces in each group and assign custom names:")

    new_labels = {}
    for person_name in st.session_state.labels:
        st.divider()
        cluster_num = int(person_name.split()[-1]) - 1
        faces_in_cluster = [fd for fd in st.session_state.face_data if fd['cluster'] == cluster_num]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"**{person_name}** ({len(faces_in_cluster)} photos)")
            preview_faces = faces_in_cluster[:10]
            cols = st.columns(min(len(preview_faces), 5))
            for idx, face_data in enumerate(preview_faces):
                with cols[idx % 5]:
                    st.image(face_data['image'], width=100, caption=f"#{idx+1}")
            if len(faces_in_cluster) > 10:
                st.caption(f"...and {len(faces_in_cluster) - 10} more")
        with col2:
            st.write("")
            st.write("")
            new_name = st.text_input(
                f"Rename {person_name}", 
                value=person_name, 
                key=f"rename_{person_name}",
                placeholder="Enter name..."
            )
            new_labels[person_name] = new_name

    if st.button("💾 Apply Name Changes", type="primary"):
        new_counts = {}
        for old_name, new_name in new_labels.items():
            if old_name in st.session_state.counts:
                new_counts[new_name] = st.session_state.counts[old_name]
        st.session_state.counts = new_counts
        st.session_state.labels = list(new_counts.keys())
        st.success("✅ Names updated successfully!")
        st.rerun()

if st.session_state.counts:
    st.divider()
    st.subheader("📊 Photo Count per Person")
    df = pd.DataFrame({
        "Name": list(st.session_state.counts.keys()),
        "Count": list(st.session_state.counts.values())
    }).sort_values("Count", ascending=False)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.metric("Total Faces", df["Count"].sum())
        st.metric("Unique People", len(df))
    with col2:
        chart = alt.Chart(df).mark_bar(color='#1f77b4').encode(
            x=alt.X("Count:Q", title="Number of Photos"),
            y=alt.Y("Name:N", sort="-x", title="Person"),
            tooltip=["Name", "Count"]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

if st.session_state.counts:
    st.subheader("💾 Export Results")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="face_count_report.csv",
        mime="text/csv"
    )

st.divider()
if st.button("🔄 Reset Session", type="secondary"):
    for key in ["face_data", "labels", "counts", "files", "processed", "cluster_labels"]:
        st.session_state.pop(key, None)
    st.rerun()
