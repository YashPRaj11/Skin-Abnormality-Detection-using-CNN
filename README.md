# 🧠 Skin Abnormality Detection Using CNN

A lightweight deep-learning application for **binary skin-image classification** using a Convolutional Neural Network (CNN). The project includes an image preprocessing/training pipeline and a Streamlit interface for testing the trained model on uploaded images.

> **Note:** This project is intended for educational/research purposes and is **not a medical diagnostic tool**.

---

## 📌 Overview

The system classifies an uploaded skin image into one of two categories:

- **Normal Skin**
- **Abnormal Skin**

The application accepts JPG, JPEG, and PNG images, preprocesses them to the model's required input size, and uses a trained Keras model to generate a prediction.

The Streamlit application resizes uploaded images to **224 × 224 pixels**, converts them to RGB, scales pixel values to the `[0, 1]` range, and passes the image to the trained model. The model uses a sigmoid output and a **0.5 decision threshold** for the displayed classification.

---

## ✨ Features

- Binary skin-image classification
- CNN-based image classification
- Image resizing to `224 × 224`
- RGB image preprocessing
- Pixel normalization
- Keras `.keras` model support
- Streamlit web interface
- Upload support for:
  - `.jpg`
  - `.jpeg`
  - `.png`
- Lightweight model suitable for experimentation and deployment

---

## 🏗️ Project Structure

```text
Skin-Abnormality-Detection/
│
├── app.py
├── Image_preprocessing.ipynb
├── skin.keras
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application for loading the trained model and performing predictions |
| `Image_preprocessing.ipynb` | Dataset preprocessing, splitting, model construction, compilation, and training workflow |
| `skin.keras` | Saved trained Keras classification model |
| `requirements.txt` | Python dependencies required by the application |
| `README.md` | Project documentation |

---

## 🔬 Methodology

The project follows the following workflow:

```text
Raw Dataset
     │
     ▼
Image Resizing
224 × 224
     │
     ▼
Train / Validation / Test Split
     │
     ▼
TensorFlow Dataset Loading
     │
     ▼
CNN Training
     │
     ▼
Saved Keras Model
     │
     ▼
Streamlit Application
     │
     ▼
Uploaded Skin Image
     │
     ▼
Prediction
     │
     ▼
Normal / Abnormal
```

---

## 🧹 Data Preprocessing

The notebook preprocesses the image dataset before training.

### 1. Image Resizing

Images are resized to:

```text
224 × 224
```

OpenCV is used for resizing with area interpolation.

### 2. Dataset Splitting

The dataset is divided into:

- **70% Training**
- **15% Validation**
- **15% Testing**

The split is performed separately for the `normal` and `abnormal` classes using `train_test_split` with `random_state=42`.

### 3. TensorFlow Dataset

The images are loaded using:

```python
tf.keras.utils.image_dataset_from_directory()
```

with:

```text
image_size = (224, 224)
batch_size = 32
label_mode = "binary"
```

The datasets are also prefetched using TensorFlow `AUTOTUNE`.

---

## 🧠 CNN Architecture

The training notebook defines a CNN consisting of four convolutional blocks followed by fully connected layers.

```text
Input: 224 × 224 × 3
        │
        ▼
Rescaling (1/255)
        │
        ▼
Conv2D (16 filters) + MaxPooling
        │
        ▼
Conv2D (32 filters) + MaxPooling
        │
        ▼
Conv2D (64 filters) + MaxPooling
        │
        ▼
Conv2D (128 filters) + MaxPooling
        │
        ▼
Flatten
        │
        ▼
Dense (128, ReLU)
        │
        ▼
Dropout (0.5)
        │
        ▼
Dense (1, Sigmoid)
```

The model is compiled with:

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy

The notebook configures training for up to **50 epochs**.

---

## 💾 Saved Model

The repository includes a trained model:

```text
skin.keras
```

The saved model uses a compact CNN architecture with:

- Input size: `224 × 224 × 3`
- Three convolutional layers
- Max-pooling after each convolutional layer
- Flatten layer
- Dense layer with 16 neurons
- Sigmoid output layer

The model is loaded by the Streamlit application using:

```python
model = load_model("skin.keras")
```

---

## 🖥️ Streamlit Application

The application is implemented in `app.py`.

### Prediction Pipeline

When an image is uploaded:

1. The image is opened with PIL.
2. It is converted to RGB.
3. It is resized to `224 × 224`.
4. Pixel values are divided by `255.0`.
5. A batch dimension is added.
6. The image is passed to the CNN.
7. The sigmoid prediction is evaluated using a `0.5` threshold.

Conceptually:

```text
Uploaded Image
      ↓
RGB Conversion
      ↓
224 × 224 Resize
      ↓
Pixel Normalization
      ↓
CNN Model
      ↓
Sigmoid Prediction
      ↓
0.5 Threshold
      ↓
Normal / Abnormal
```

---

## ⚙️ Requirements

The project uses the following packages:

```text
streamlit
tensorflow==2.19.0
keras==3.13.2
numpy
pillow
h5py
gdown
opencv-python
opencv-python-headless
```

These dependencies are listed in `requirements.txt`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Using the Application

1. Start the Streamlit application.
2. Click **Choose an image...**
3. Upload a `.jpg`, `.jpeg`, or `.png` skin image.
4. The uploaded image will be displayed.
5. The model will generate a prediction.
6. The application displays either:
   - `Abnormal Skin Detected`
   - `Normal Skin`

---

## 📊 Prediction Logic

The application obtains the first value from the model's sigmoid output:

```python
prediction = model.predict(processed_image)[0][0]
```

It then applies:

```python
if prediction < 0.5:
    # Abnormal
else:
    # Normal
```

The displayed score is formatted to two decimal places.

---

## 🔧 Customization

### Change the Decision Threshold

The classification threshold can be modified in `app.py`:

```python
if prediction < 0.5:
```

For example:

```python
if prediction < 0.4:
```

Changing the threshold changes the operating point of the classifier and should be evaluated using an appropriate validation/test set.

### Change the Input Image Size

The application currently uses:

```python
img_size = (224, 224)
```

This should remain consistent with the input dimensions used when training the model.

---

## ⚠️ Important Notes

### Model and Training Architecture

The training notebook and the supplied `skin.keras` file do not have identical CNN configurations. The notebook defines a four-convolutional-layer model, while inspection of the saved model shows a compact three-convolutional-layer architecture.

Therefore, if retraining the project from the notebook, do not assume that the resulting model will be identical to the supplied `skin.keras` file.

### Keras Version

The supplied model contains Keras metadata indicating it was saved with **Keras 3.10.0**, while the current `requirements.txt` specifies `keras==3.13.2`.

If model-loading compatibility issues occur, verify the TensorFlow/Keras version combination used to save and load the model.

### Medical Use

This project is a machine-learning research/educational implementation. It should not be used as a substitute for examination or diagnosis by a qualified healthcare professional.

---

## 📚 Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **NumPy**
- **Pillow**
- **Streamlit**
- **Scikit-learn**

---

## 🔮 Possible Future Improvements

- Add data augmentation during training
- Add class-wise precision, recall, F1-score, and confusion matrix
- Evaluate the model on an independent external dataset
- Add Grad-CAM or other explainability methods
- Improve class balancing if required
- Add model confidence visualization
- Compare the custom CNN with transfer-learning architectures
- Optimize the model for mobile/edge deployment
- Add model versioning and experiment tracking

---

## 📄 License

Add the license appropriate for your repository.

---

## 👨‍💻 Author

**Yash P. Raj**

B.Tech — Artificial Intelligence & Data Science

GitHub: `https://github.com/YashPRaj11`
