# app.py - IMPROVED VERSION
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
import io
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import dlib

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
st.write("Upload images. Faces will be automatically grouped; assign custom names with preview.")

# Load face detection and recognition models
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Load dlib's face recognition model for better feature extraction
    try:
        face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        dlib_detector = dlib.get_frontal_face_detector()
        use_dlib = True
    except:
        st.warning("⚠️ Dlib models not found. Using basic mode. For better accuracy, download dlib models.")
        face_rec_model = None
        shape_predictor = None
        dlib_detector = None
        use_dlib = False
    
    return face_cascade, eye_cascade, face_rec_model, shape_predictor, dlib_detector, use_dlib

face_detector, eye_detector, face_rec_model, shape_predictor, dlib_detector, use_dlib = load_models()

def is_valid_face(face_roi_gray, face_w, face_h, eyes):
    """Enhanced validation to reduce false positives"""
    
    # Check 1: Must detect at least one eye
    if len(eyes) == 0:
        return False
    
    # Check 2: Aspect ratio should be close to 1 (faces are roughly square)
    aspect_ratio = face_w / face_h
    if aspect_ratio < 0.7 or aspect_ratio > 1.3:
        return False
    
    # Check 3: Face region should have reasonable variance (not uniform like clothing)
    variance = np.var(face_roi_gray)
    if variance < 200:  # Too uniform, likely not a face
        return False
    
    # Check 4: Check if eyes are in upper half of face
    for (ex, ey, ew, eh) in eyes:
        if ey > face_h * 0.6:  # Eyes too low, likely false positive
            return False
    
    return True

def extract_face_features_improved(face_roi_gray, face_roi_color, dlib_img=None, rect=None):
    """Improved feature extraction that works better across angles"""
    
    # Method 1: Use dlib if available (much better for angles)
    if use_dlib and dlib_img is not None and rect is not None:
        try:
            shape = shape_predictor(dlib_img, rect)
            face_descriptor = np.array(face_rec_model.compute_face_descriptor(dlib_img, shape))
            return face_descriptor
        except:
            pass
    
    # Method 2: Enhanced traditional features
    face_resized = cv2.resize(face_roi_gray, (128, 128))
    
    # Apply histogram equalization for better lighting normalization
    face_resized = cv2.equalizeHist(face_resized)
    
    # 1. Raw pixel features
    pixels = face_resized.flatten().astype(float) / 255.0
    
    # 2. Histogram features (more robust to angle changes)
    hist = cv2.calcHist([face_resized], [0], None, [64], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-7)
    
    # 3. HOG features (better for different angles)
    win_size = (128, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(face_resized)
    hog_features = hog_features.flatten()
    hog_features = hog_features / (np.linalg.norm(hog_features) + 1e-7)
    
    # 4. LBP features (rotation invariant)
    lbp_features = []
    for i in range(1, face_resized.shape[0]-1):
        for j in range(1, face_resized.shape[1]-1):
            center = face_resized[i, j]
            code = 0
            code |= (face_resized[i-1, j-1] > center) << 7
            code |= (face_resized[i-1, j] > center) << 6
            code |= (face_resized[i-1, j+1] > center) << 5
            code |= (face_resized[i, j+1] > center) << 4
            code |= (face_resized[i+1, j+1] > center) << 3
            code |= (face_resized[i+1, j] > center) << 2
            code |= (face_resized[i+1, j-1] > center) << 1
            code |= (face_resized[i, j-1] > center) << 0
            lbp_features.append(code)
    
    lbp_hist, _ = np.histogram(lbp_features, bins=32, range=(0, 256))
    lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-7)
    
    # Combine all features (weighted combination)
    features = np.concatenate([
        pixels * 0.3,      # Reduced weight for raw pixels
        hist * 0.2,
        hog_features[:256] * 0.3,  # Increased weight for HOG
        lbp_hist * 0.2
    ])
    
    return features

# --- File uploader ---
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
    value=3,  # Lower default for better angle handling
    step=1,
    help="Lower values group similar faces together (better for same person at different angles). Higher values create more separate groups."
)

# --- Process images ---
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
        
        # Read image
        img_bytes = file.read()
        file.seek(0)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray_eq = cv2.equalizeHist(gray)
        
        # Detect faces with STRICTER parameters to reduce false positives
        faces = face_detector.detectMultiScale(
            gray_eq, 
            scaleFactor=1.1,    # More careful scaling
            minNeighbors=7,     # INCREASED from 4 to 7 - reduces false positives significantly
            minSize=(80, 80),   # Larger minimum size - avoids detecting small objects
            maxSize=(500, 500), # Maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Try dlib detector if available (better for angles)
        if use_dlib:
            dlib_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dlib_faces = dlib_detector(dlib_img, 1)
        
        # Extract face features with validation
        for face_idx, (x, y, w, h) in enumerate(faces):
            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = image[y:y+h, x:x+w]
            
            # Detect eyes for validation
            eyes = eye_detector.detectMultiScale(
                face_roi_gray, 
                scaleFactor=1.1, 
                minNeighbors=5,  # INCREASED from 3 to 5 - stricter eye detection
                minSize=(20, 20)
            )
            
            # ENHANCED VALIDATION - reduces false positives
            if not is_valid_face(face_roi_gray, w, h, eyes):
                continue  # Skip this detection
            
            # Extract features
            rect = None
            if use_dlib and dlib_faces:
                # Find matching dlib face
                for dlib_rect in dlib_faces:
                    dx, dy, dw, dh = dlib_rect.left(), dlib_rect.top(), dlib_rect.width(), dlib_rect.height()
                    # Check if this dlib face overlaps with current opencv face
                    if abs(x - dx) < 50 and abs(y - dy) < 50:
                        rect = dlib_rect
                        break
            
            features = extract_face_features_improved(face_roi_gray, face_roi_color, dlib_img if use_dlib else None, rect)
            all_features.append(features)
            
            # Store face image
            face_rgb = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            all_face_images.append(face_pil)
        
        progress.progress((i+1)/n)
    
    status.text("🔍 Grouping similar faces...")
    
    if all_features:
        X = np.array(all_features)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # ADJUSTED clustering for better angle handling
        distance_threshold = sensitivity * 3.5  # Reduced multiplier for tighter grouping
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='ward'  # Changed from 'average' to 'ward' - better for face clustering
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

# [REST OF THE CODE REMAINS THE SAME - Name editing, reports, etc.]
# --- Name editing with preview ---
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

# --- Report & Visualization ---
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
