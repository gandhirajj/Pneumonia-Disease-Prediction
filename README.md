Pneumonia Detection Using CNN + LSTM Hybrid Model

This project implements a deep learning model to detect pneumonia from pediatric chest X-ray images using a hybrid architecture that combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks. The model is trained using TensorFlow/Keras and evaluated on test data with visual metrics and performance reports.

---

📂 Dataset

We use the Pediatric Chest X-ray Dataset, which contains labeled chest X-ray images of pediatric patients categorized as:

- Normal
- Pneumonia

> 📌 Folder Structure:
Pediatric Chest X-ray Pneumonia/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/


> Dataset Source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray)

---

🧠 Deep Learning Architectures

🌀 Convolutional Neural Networks (CNNs)

CNNs are powerful deep learning models designed specifically for image data. They apply convolutional filters to input images to extract spatial features such as edges, textures, and patterns.

Key Components:
- Conv2D layers: Apply filters that learn to recognize visual features.
- MaxPooling2D: Downsample feature maps to reduce dimensionality and computational cost.
- Activation Functions: Commonly ReLU, which introduces non-linearity.

Why CNN?  
CNNs are excellent at learning spatial hierarchies in image data, which is essential for understanding medical images like chest X-rays.

---

🔁 Long Short-Term Memory (LSTM)

LSTMs are a type of Recurrent Neural Network (RNN) that excel at learning long-term dependencies in sequential data. Unlike standard RNNs, LSTMs solve the vanishing gradient problem and remember patterns over longer sequences.

Key Components:
- Memory Cell: Maintains a memory of previous states.
- Gates (Forget, Input, Output): Control the flow of information.

Why LSTM in Image Tasks?
Although LSTMs are primarily used for sequences like text or time series, they can also be applied to image data when the image is reshaped into a sequence (e.g., by treating rows or feature vectors as timesteps). This allows the model to learn spatial sequences across image features.

---

🔗 CNN + LSTM Hybrid Model

This architecture combines CNN and LSTM to leverage the best of both worlds:

1. CNN extracts spatial features from X-ray images.
2. Reshape Layer converts the CNN output into a sequence.
3. LSTM processes the sequence to learn temporal/spatial dependencies across feature maps.
4. Dense Layers make the final classification.

Advantages:
- CNN captures local features like shapes and textures.
- LSTM captures relationships across regions, such as the spread of pneumonia patterns in the lungs.



🔍 1. Image Classification
Concept:
Classifying an image into predefined categories (here: Normal vs Pneumonia).

Requires understanding the spatial patterns and features present in medical images (chest X-rays).



🧠 2. Convolutional Neural Networks (CNNs)
Purpose:
Automatically learn and extract spatial features from images (e.g., edges, blobs, textures).

Components Used:
Conv2D: Applies filters over the image to detect patterns.

MaxPooling2D: Downsamples feature maps, reducing size and computation.

Activation Function (ReLU): Introduces non-linearity to learn complex functions.

He Initialization: Helps with weight initialization to maintain stable gradient flow.

Why used?
CNNs are highly effective for image data and are the backbone of most visual recognition tasks.



🔁 3. Long Short-Term Memory (LSTM)
Purpose:
Learn sequential dependencies — typically in time-series or language tasks, but here it's used on reshaped image features.

Components Used:
LSTM Layer: Processes a sequence of feature vectors (produced from CNN) and captures temporal relationships across them.

Why used?
In this project, the CNN's output is reshaped into a sequence, and LSTM is used to understand long-range dependencies across different parts of the image.

It allows the model to understand contextual spatial relationships, especially helpful in detecting subtle pneumonia spread patterns.



🔗 4. CNN + LSTM Hybrid Architecture

Why combine them?
CNN: Learns local patterns like edges and shapes (important for lung structure).

LSTM: Learns global dependencies across different regions in the image (e.g., the spread of infection).

Flow:
mathematica
Copy code
Input Image (224x224x3)
   ↓
CNN Layers → Feature Maps
   ↓
Reshape (Convert 3D features to sequences)
   ↓
LSTM Layer → Context-aware understanding
   ↓
Dense Layers → Classification
   ↓
Softmax Output → Normal or Pneumonia
This architecture is especially useful when patterns in different regions of an image influence each other — which is common in medical images.



🧪 5. Model Compilation and Training

Concepts Used:
Loss Function: sparse_categorical_crossentropy for multi-class classification with integer labels.

Optimizer: Adam for adaptive learning rate.

Metrics: accuracy to track performance.

Training Details:
Images normalized (/255.0)

Model trained for 20 epochs with 20% validation split



📊 6. Evaluation Metrics

Concepts Used:
Accuracy: Percentage of correct predictions.

Confusion Matrix: Shows TP, FP, TN, FN to understand model behavior.

Classification Report: Provides precision, recall, and F1-score for each class.



🖼️ 7. Visualization with Matplotlib & Seaborn
Concepts Used:
Plotting the confusion matrix for better visual analysis.

Using seaborn.heatmap() to make it easier to interpret predictions.



🧾 Summary of Key Concepts

Concept	Purpose

CNN	                   Extract spatial features from X-ray images
LSTM	               Understand spatial dependencies across image regions
Reshape Layer	       Convert CNN output to sequence format for LSTM
Dense + Softmax	       Final classification into Normal/Pneumonia
Image Normalization	   Scale image pixels to [0,1] for stable training
Model Evaluation	   Accuracy, confusion matrix, precision, recall
Visualization	       Helps interpret model performance visually



```bash
# Clone the repository
git clone https://github.com/your-repo.git
# Navigate to the project directory
cd Pneumonia
# Install dependencies
pip install numpy tensorflow scikit-learn matplotlib seaborn
Run the Script
python app.py

