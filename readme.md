# Face Recognition & Photo Count App

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated Streamlit web application that leverages computer vision to automatically detect, group, and analyze faces in your photo collection. Perfect for organizing large photo libraries and tracking individual appearances across multiple images.

## 🎯 Overview

This application uses OpenCV's face detection algorithms combined with unsupervised clustering to identify unique individuals across your photo collection. Upload your photos, and the app will automatically group faces, allow custom labeling, and provide visual analytics on photo participation.

## ✨ Features

- **🎭 Intelligent Face Detection**: Powered by OpenCV's Haar Cascade classifiers for robust face recognition
- **👥 Smart Face Clustering**: Uses scikit-learn's clustering algorithms to group similar faces automatically
- **✏️ Custom Name Assignment**: Interactive interface for labeling and managing detected individuals
- **📊 Visual Analytics**: Beautiful charts and statistics powered by Altair for data visualization
- **💾 Export Capabilities**: Download results as CSV for further analysis or record-keeping
- **📁 Bulk Upload Support**: Process up to 1GB of images in JPG/PNG formats
- **🎨 User-Friendly Interface**: Clean, intuitive Streamlit-based UI with real-time feedback
- **📖 Built-in User Guide**: Comprehensive help documentation accessible within the app

## 🛠️ Technology Stack

- **Frontend Framework**: Streamlit
- **Computer Vision**: OpenCV (opencv-python-headless)
- **Machine Learning**: scikit-learn (clustering algorithms)
- **Data Processing**: NumPy, Pandas
- **Visualization**: Altair
- **Image Processing**: Pillow (PIL)
- **Deep Learning**: TensorFlow, Keras (tf-keras)

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- System dependencies:
  - `libgl1` (Linux) - required for OpenCV

## 🚀 Installation

### Option 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/towhidulislam1999/Face-Recognition-Photo-Count-App.git
   cd Face-Recognition-Photo-Count-App
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system dependencies** (Linux/Ubuntu):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1
   ```

### Option 2: Streamlit Cloud Deployment

This application is configured for easy deployment on Streamlit Cloud:

1. Fork this repository to your GitHub account
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy the app by selecting the repository and `app.py` as the main file
4. The `packages.txt` file will automatically install required system dependencies

## 💻 Usage

### Running the Application

1. **Start the Streamlit server**:
   ```bash
   streamlit run app.py
   ```

2. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:8501`

### Step-by-Step Workflow

1. **Upload Photos**: 
   - Click the upload button or drag-and-drop images (JPG/PNG)
   - Supports multiple file uploads up to 1GB total

2. **Adjust Settings**:
   - Use the face grouping sensitivity slider to control clustering threshold
   - Higher values create more groups (stricter matching)
   - Lower values create fewer groups (looser matching)

3. **Process Images**:
   - Click the "Process Images" button
   - Wait for face detection and clustering to complete

4. **Review and Label**:
   - Browse detected face groups
   - Assign custom names to each person
   - Edit or merge groups as needed

5. **Analyze Results**:
   - View visual analytics and statistics
   - See photo count per person
   - Export data to CSV for external use

## 📁 Project Structure

```
Face-Recognition-Photo-Count-App/
├── .streamlit/              # Streamlit configuration
│   └── config.toml         # App theming and settings
├── pages/                   # Multi-page app components
│   └── User_Guide.py       # Built-in documentation
├── app.py                   # Main application file
├── requirements.txt         # Python dependencies
├── packages.txt            # System dependencies (for Streamlit Cloud)
├── .gitignore              # Git ignore rules
├── .python-version.11      # Python version specification
└── readme.md               # This file
```

## 🔧 Configuration

### Streamlit Configuration

Customize the app's appearance and behavior by editing `.streamlit/config.toml`:
- Theme colors
- Upload size limits
- Server settings

### Face Detection Parameters

Adjust detection sensitivity in the application's sidebar:
- **Clustering Threshold**: Controls how strictly faces are matched
- **Minimum Face Size**: Filter out very small detections

## 📊 Dependencies

### Python Packages (requirements.txt)

```
streamlit>=1.30.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
altair>=5.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
tf-keras>=2.15.0
```

### System Packages (packages.txt)

```
libgl1
```

## 🐛 Troubleshooting

### Common Issues

**Problem**: `ImportError: libGL.so.1: cannot open shared object file`
- **Solution**: Install libgl1 system package
  ```bash
  sudo apt-get install -y libgl1
  ```

**Problem**: Face detection is too slow
- **Solution**: 
  - Reduce image resolution before uploading
  - Process fewer images at once
  - Use opencv-python instead of opencv-python-headless for GPU acceleration (requires display)

**Problem**: Too many/too few face groups
- **Solution**: Adjust the face grouping sensitivity slider
  - Increase for stricter matching (more groups)
  - Decrease for looser matching (fewer groups)

**Problem**: Upload limit exceeded
- **Solution**: 
  - Process photos in smaller batches
  - Adjust `server.maxUploadSize` in `.streamlit/config.toml`

**Problem**: Application crashes during processing
- **Solution**:
  - Check available system memory
  - Reduce batch size
  - Ensure all dependencies are correctly installed

## ❓ FAQ

**Q: What image formats are supported?**
A: JPG/JPEG and PNG formats are fully supported.

**Q: Is my photo data stored or transmitted?**
A: No. All processing happens locally in your browser session. Photos are not stored or transmitted to any external servers.

**Q: Can I use this for commercial purposes?**
A: Yes, this project is licensed under MIT License, allowing commercial use with attribution.

**Q: How accurate is the face detection?**
A: Accuracy depends on image quality, lighting, and face angles. Best results are achieved with front-facing, well-lit photos.

**Q: Can it detect faces with masks or sunglasses?**
A: Detection accuracy may be reduced for partially obscured faces. The algorithm works best with clearly visible faces.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
3. **Make your changes** and commit:
   ```bash
   git commit -m "Add: Your feature description"
   ```
4. **Push to your branch**:
   ```bash
   git push origin feature/YourFeatureName
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines for Python code
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting
- Keep commits atomic and well-described

### Areas for Contribution

- 🎨 UI/UX improvements
- 🚀 Performance optimizations
- 🧪 Additional face detection algorithms
- 📝 Documentation enhancements
- 🌐 Internationalization (i18n)
- 🧪 Unit tests and integration tests

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Sublicense

With the requirement to:
- 📄 Include original license and copyright notice

## 👨‍💻 Author

**Towhidul Islam**
- GitHub: [@towhidulislam1999](https://github.com/towhidulislam1999)
- Repository: [Face-Recognition-Photo-Count-App](https://github.com/towhidulislam1999/Face-Recognition-Photo-Count-App)

## 🙏 Acknowledgments

- **Streamlit Team** - For the amazing web app framework
- **OpenCV Contributors** - For robust computer vision libraries
- **scikit-learn Team** - For machine learning algorithms
- **Open Source Community** - For continuous inspiration and support

## 📚 Related Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Face Detection with OpenCV](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

## 🔮 Future Enhancements

- [ ] Support for video file processing
- [ ] Advanced face recognition with deep learning models
- [ ] Export results as PDF reports
- [ ] Database integration for persistent storage
- [ ] Multi-user support with authentication
- [ ] Mobile-responsive design improvements
- [ ] Real-time webcam face detection
- [ ] Integration with cloud storage services

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ using Streamlit and OpenCV
