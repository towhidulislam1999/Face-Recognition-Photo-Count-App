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

def extract_faces_and_embeddings(image_bgr):
    """
    Extract faces using DeepFace's built-in detection (more accurate than Haar Cascade).
    Returns list of (face_image_rgb, embedding) tuples.
    """
    results = []
    try:
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Use DeepFace to detect and extract faces with embeddings
        # This uses MTCNN detector which is much more accurate than Haar Cascade
        detections = DeepFace.extract_faces(
            img_path=image_rgb,
            detector_backend='retinaface',  # More accurate than opencv
            enforce_detection=False,
            align=True
        )
        
        for detection in detections:
            # Get face region
            face_array = detection['face']
            
            # Check confidence score to filter false positives
            confidence = detection.get('confidence', 0)
            
            # Only include high-confidence detections
            if confidence > 0.90:  # 90% confidence threshold
                # Convert face to uint8 if needed
                if face_array.max() <= 1.0:
                    face_array = (face_array * 255).astype(np.uint8)
                
                # Get embedding
                try:
                    embedding_result = DeepFace.represent(
                        img_path=face_array,
                        model_name='Facenet',
                        enforce_detection=False
                    )
                    embedding = np.array(embedding_result[0]['embedding'])
                    
                    # Additional validation: check face quality
                    # Calculate face sharpness (Laplacian variance)
                    if len(face_array.shape) == 3:
                        gray_face = cv2.cvtColor(face_array, cv2.COLOR_RGB2GRAY)
                    else:
                        gray_face = face_array
                    
                    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    
                    # Only include if face is reasonably sharp (not blurry pattern)
                    if laplacian_var > 50:  # Threshold for blur detection
                        results.append((face_array, embedding))
                        
                except Exception as e:
                    continue
                    
    except Exception as e:
        # If DeepFace fails completely, return empty list
        pass
    
    return results

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
    value=4, 
    step=1,
    help="Lower values group similar faces together. Higher values create more separate groups."
)

# --- Detection quality filter ---
st.sidebar.header("🔧 Advanced Settings")
confidence_threshold = st.sidebar.slider(
    "Face Detection Confidence",
    min_value=0.80,
    max_value=0.99,
    value=0.90,
    step=0.01,
    help="Higher values = fewer false positives (clothes, patterns) but might miss some faces"
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
    
    # Statistics
    total_detections = 0
    filtered_detections = 0

    for i, file in enumerate(st.session_state.files):
        status.text(f"Processing image {i+1}/{n}: {file.name}")

        try:
            img_bytes = file.read()
            file.seek(0)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Extract faces with embeddings using improved detection
            face_results = extract_faces_and_embeddings(image)
            
            for face_array, embedding in face_results:
                total_detections += 1
                all_features.append(embedding)
                face_pil = Image.fromarray(face_array)
                all_face_images.append(face_pil)
                filtered_detections += 1
                
        except Exception as e:
            st.sidebar.warning(f"Error processing {file.name}: {str(e)}")
            continue
            
        progress.progress((i+1)/n)

    status.text("🔍 Grouping similar faces...")

    if all_features:
        X = np.array(all_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adjusted distance threshold for Facenet embeddings
        distance_threshold = sensitivity * 1.2
        
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
        
        st.success(f"✅ Processing complete! Found {len(all_features)} valid faces in {len(unique_labels)} groups.")
        
        if total_detections > filtered_detections:
            st.info(f"ℹ️ Filtered out {total_detections - filtered_detections} low-quality detections (likely false positives)")
    else:
        status.empty()
        st.warning("⚠️ No faces detected. Try different images or adjust the confidence threshold in the sidebar.")

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
        # Create mapping from old person names to new names
        name_mapping = {}
        for old_name, new_name in new_labels.items():
            cluster_num = int(old_name.split()[-1]) - 1
            name_mapping[cluster_num] = new_name
        
        # Update cluster assignments in face_data based on new names
        # Group clusters with same name together
        name_to_clusters = {}
        for cluster_num, name in name_mapping.items():
            if name not in name_to_clusters:
                name_to_clusters[name] = []
            name_to_clusters[name].append(cluster_num)
        
        # Reassign cluster labels for merged groups
        new_cluster_assignment = {}
        for name, cluster_list in name_to_clusters.items():
            # All clusters with same name get the same new cluster number
            primary_cluster = min(cluster_list)
            for cluster in cluster_list:
                new_cluster_assignment[cluster] = primary_cluster
        
        # Update face_data with merged clusters
        for face_data in st.session_state.face_data:
            old_cluster = face_data['cluster']
            face_data['cluster'] = new_cluster_assignment.get(old_cluster, old_cluster)
        
        # Recalculate counts based on merged groups
        new_counts = {}
        for name, cluster_list in name_to_clusters.items():
            # Count all faces in all clusters with this name
            total_count = sum([
                len([fd for fd in st.session_state.face_data if fd['cluster'] == new_cluster_assignment.get(c, c)])
                for c in cluster_list
            ])
            # Use set to avoid counting same cluster multiple times
            merged_cluster = new_cluster_assignment.get(cluster_list[0], cluster_list[0])
            face_count = len([fd for fd in st.session_state.face_data if fd['cluster'] == merged_cluster])
            new_counts[name] = face_count
        
        st.session_state.counts = new_counts
        st.session_state.labels = list(new_counts.keys())
        
        # Show merge information
        merged_groups = [name for name, clusters in name_to_clusters.items() if len(clusters) > 1]
        if merged_groups:
            st.success(f"✅ Names updated! Merged groups: {', '.join(merged_groups)}")
        else:
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
    for key in ["face_data", "labels", "counts", "files", "processed", "cluster_labels"]:
        st.session_state.pop(key, None)
    st.rerun()