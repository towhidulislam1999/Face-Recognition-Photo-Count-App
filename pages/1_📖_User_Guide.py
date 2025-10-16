# pages/1_📖_User_Guide.py
import streamlit as st

st.set_page_config(
    page_title="User Guide",
    page_icon="📖"
)

st.title("📖 User Guide")
st.write("Learn how to use the Face Count App effectively")

st.divider()

# Step-by-step instructions
st.subheader("🚀 Getting Started")

with st.expander("**Step 1: Upload Your Images**", expanded=True):
    st.write("""
    - Click the **"Upload images"** button on the main page
    - Select multiple images (JPG, PNG formats supported)
    - You can drag and drop files or select from folders
    - Maximum upload size: 1 GB total
    """)

with st.expander("**Step 2: Adjust Sensitivity Slider**"):
    st.write("""
    The sensitivity slider controls how faces are grouped together:
    
    **Lower values (1-3):**
    - Groups more faces together
    - Use when you have many duplicate faces of the same people
    - Best for family photos or events with the same attendees
    
    **Medium values (4-6):**
    - Balanced grouping (recommended default)
    - Good for mixed scenarios
    
    **Higher values (7-10):**
    - Creates more separate groups
    - Use when faces are very different
    - Reduces false matches but may split the same person into multiple groups
    """)

with st.expander("**Step 3: Process Images**"):
    st.write("""
    1. Click the **"🚀 Process Images"** button
    2. Wait for the progress bar to complete
    3. The app will detect and group similar faces automatically
    4. You'll see a summary of how many faces and groups were found
    """)

with st.expander("**Step 4: Review Face Previews**"):
    st.write("""
    - Each person group shows preview thumbnails
    - Up to 5 face samples are displayed per group
    - Check if faces are grouped correctly
    - If grouping is wrong, adjust sensitivity and reprocess
    """)

with st.expander("**Step 5: Rename Persons**"):
    st.write("""
    - Look at the face previews for each group
    - Enter custom names in the text boxes (e.g., "John", "Sarah")
    - Replace default names like "Person 1" with actual names
    - Click **"💾 Apply Name Changes"** when done
    """)

with st.expander("**Step 6: View Report & Export**"):
    st.write("""
    - See a table showing photo counts per person
    - View the bar chart visualization
    - Download CSV report for your records
    - Use the reset button to start over with new images
    """)

st.divider()

# Tips and tricks
st.subheader("💡 Tips for Best Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **For Better Face Detection:**
    - Use clear, well-lit photos
    - Ensure faces are visible (not covered/blurred)
    - Front-facing photos work best
    - Minimum face size: 50x50 pixels
    """)

with col2:
    st.markdown("""
    **For Accurate Grouping:**
    - Start with default sensitivity (4)
    - If same person split → lower sensitivity
    - If different people merged → raise sensitivity
    - Reprocess after adjusting slider
    """)

st.divider()

# Troubleshooting
st.subheader("🔧 Troubleshooting")

st.markdown("""
| Issue | Solution |
|-------|----------|
| No faces detected | Try different images with clearer faces, or adjust lighting |
| Same person in multiple groups | Lower the sensitivity slider (try 2-3) and reprocess |
| Different people grouped together | Raise the sensitivity slider (try 7-8) and reprocess |
| Upload limit exceeded | Remove some files or process in smaller batches |
| Slow processing | Large images take longer; consider resizing before upload |
""")

st.divider()

# Technical details
with st.expander("🔬 Technical Details (Optional)"):
    st.write("""
    **Face Detection:** Uses OpenCV's Haar Cascade classifier with eye detection for verification
    
    **Face Grouping:** Agglomerative Clustering algorithm with normalized feature vectors
    
    **Features Extracted:** Raw pixel data (128x128) + histogram features (32 bins)
    
    **Sensitivity Mapping:** Slider value × 5 = distance threshold for clustering
    """)

st.divider()

st.info("💬 **Need Help?** Go back to the main app to start processing your images!")
