# Face Recognition & Photo Count App

A Streamlit web application that automatically detects and groups faces in your photo collection, allowing you to count how many photos each person appears in.

## Features

- 🎭 Automatic face detection using OpenCV
- 👥 Smart face grouping and clustering
- ✏️ Custom name assignment for each person
- 📊 Visual analytics with charts
- 💾 Export results to CSV
- 📁 Support for bulk image uploads (up to 1GB)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### How to Use:
1. Upload your photos (JPG/PNG format)
2. Adjust the face grouping sensitivity slider
3. Click "Process Images"
4. Edit person names as needed
5. View the analytics and export results

## Requirements

- Python 3.7+
- streamlit
- opencv-python-headless
- numpy
- pandas
- altair
- pillow
- scikit-learn

## License

MIT License - feel free to use and modify!

## Author

Your Name

## Acknowledgments

Built with Streamlit and OpenCV