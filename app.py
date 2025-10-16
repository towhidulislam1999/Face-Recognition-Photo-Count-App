# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import io
from sklearn.cluster import DBSCAN

# --- Initialize session state ---
if "face_features" not in st.session_state:
    st.session_state.face_features = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "files" not in st.session_state:
    st.session_state.files = []
if "processed" not in st.session_state:
    st.session_state.processed = False

st.set_page_config(page_title="Face Count App", layout="wide")
st.title("🎭 Face Recognition & Photo Count")
st.write("Upload up to 1 GB of images. Faces will be automatically grouped; assign custom names.")

# Load face detection model
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

face_detector = load_face_detector()

# --- File uploader with size limit ---
uploaded = st.file_uploader(
    "📁 Upload images (jpg/png), multiple files or drag-and-drop folder",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded:
    # Add new files to session state
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

# Show current file count
if st.session_state.files:
    st.info(f"📊 Currently loaded: {len(st.session_state.files)} images")

# --- Sensitivity slider ---
sensitivity = st.slider(
    "🎚️ Face grouping sensitivity (lower = stricter matching)", 
    min_value=0.3, 
    max_value=0.9, 
    value=0.5, 
    step=0.05,
    help="Lower values group only very similar faces together. Higher values are more lenient."
)

# --- Process images ---
if st.button("🚀 Process Images", type="primary") and st.session_state.files:
    st.session_state.face_features = []
    st.session_state.counts = {}
    
    progress = st.progress(0)
    status = st.empty()
    n = len(st.session_state.files)
    
    all_faces = []
    
    for i, file in enumerate(st.session_state.files):
        status.text(f"Processing image {i+1}/{n}: {file.name}")
        
        # Read image
        img_bytes = file.read()
        file.seek(0)  # Reset file pointer
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Extract face features
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))
            face_vector = face_resized.flatten().astype(float) / 255.0
            all_faces.append(face_vector)
        
        progress.progress((i+1)/n)
    
    status.text("🔍 Grouping faces...")
    
    if all_faces:
        # Cluster faces using DBSCAN
        X = np.array(all_faces)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering
        eps = sensitivity * 10  # Scale sensitivity to appropriate epsilon range
        clustering = DBSCAN(eps=eps, min_samples=1, metric='euclidean').fit(X_scaled)
        
        # Count faces per cluster
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            if label == -1:
                person_name = "Unknown"
            else:
                person_name = f"Person {label + 1}"
            
            count = list(clustering.labels_).count(label)
            st.session_state.counts[person_name] = count
        
        # Store for later use
        st.session_state.labels = list(st.session_state.counts.keys())
        st.session_state.processed = True
        
        status.empty()
        st.success(f"✅ Processing complete! Found {len(all_faces)} faces in {len(unique_labels)} groups.")
    else:
        status.empty()
        st.warning("⚠️ No faces detected in the uploaded images. Try different images or adjust sensitivity.")

# --- Name editing ---
if st.session_state.processed and st.session_state.labels:
    st.subheader("✏️ Edit Person Names")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**Current Name**")
    with col2:
        st.write("**New Name**")
    
    new_labels = {}
    for label in st.session_state.labels:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text(label)
        with col2:
            new_name = st.text_input(
                f"Rename {label}", 
                value=label, 
                key=f"rename_{label}",
                label_visibility="collapsed"
            )
            new_labels[label] = new_name
    
    # Apply renaming
    if st.button("💾 Apply Name Changes"):
        new_counts = {}
        for old_name, new_name in new_labels.items():
            if old_name in st.session_state.counts:
                new_counts[new_name] = st.session_state.counts[old_name]
        
        st.session_state.counts = new_counts
        st.session_state.labels = list(new_counts.keys())
        st.success("✅ Names updated successfully!")
        st.rerun()

# --- Report & Visualization ---
if st.session_state.counts:
    st.subheader("📊 Photo Count per Person")
    
    df = pd.DataFrame({
        "Name": list(st.session_state.counts.keys()),
        "Count": list(st.session_state.counts.values())
    }).sort_values("Count", ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary stats
        st.metric("Total Faces", df["Count"].sum())
        st.metric("Unique People", len(df))
    
    with col2:
        chart = alt.Chart(df).mark_bar(color='#1f77b4').encode(
            x=alt.X("Count:Q", title="Number of Photos"),
            y=alt.Y("Name:N", sort="-x", title="Person"),
            tooltip=["Name", "Count"]
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)

# --- Export results ---
if st.session_state.counts:
    st.subheader("💾 Export Results")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Report",
        data=csv,
        file_name="face_count_report.csv",
        mime="text/csv"
    )

# --- Reset session ---
st.divider()
if st.button("🔄 Reset Session", type="secondary"):
    for key in ["face_features", "labels", "counts", "files", "processed"]:
        st.session_state.pop(key, None)
    st.rerun()