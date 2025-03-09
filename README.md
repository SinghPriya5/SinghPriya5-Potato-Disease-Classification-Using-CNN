# 🌿 🥔 Potato Disease Detection  🥔🌿

## Overview
This project aims to detect and classify potato diseases using deep learning techniques. The model is trained on an image dataset of potato leaves to identify different types of diseases and provide accurate classification results. By leveraging advanced deep learning architectures, this system helps farmers and agricultural experts detect diseases early, leading to better crop management.

## Features
✅ Image-based disease classification  
✅ Convolutional Neural Networks (CNNs) for feature extraction  
✅ High accuracy with optimized deep learning models (Achieved **98% Accuracy**)  
✅ Real-time prediction capability  
✅ Web-based interface for user-friendly interaction  
✅ Data augmentation for improved model generalization  
✅ Predicted image visualization for better insights  

## Dataset
The dataset consists of images of healthy and diseased potato leaves. The images are preprocessed, normalized, and augmented to improve model performance. It includes common potato diseases such as:

Dataset link:-  [Dataset](https://github.com/SinghPriya5/SinghPriya5-Potato-Disease-Classification-Using-CNN/tree/main/PlantVillage)

- **Early Blight**
- **Late Blight**
- **Healthy Leaves**

## Technologies Used
🚀 Python  
🚀 TensorFlow / Keras  
🚀 OpenCV  
🚀 NumPy, Pandas  
🚀 Matplotlib, Seaborn  
🚀 Flask (for deployment)  
🚀 Jupyter Notebook (for model training and evaluation)  

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/potato-disease-detection.git
   cd potato-disease-detection
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```
4. Start the Flask app for web-based predictions:
   ```sh
   python app.py
   ```

## Model Architecture
🔹 **Convolutional Layers:** Extract spatial features  
🔹 **Pooling Layers:** Reduce dimensionality  
🔹 **Fully Connected Layers:** Perform classification  
🔹 **Softmax Activation:** Outputs multi-class probability distribution  

## Evaluation Metrics
📊 **Accuracy:** 98%  
📊 **Precision, Recall, F1-score**  
📊 **Confusion Matrix for detailed performance insights**  

## Predicted Image Visualization
After making predictions, the system provides a visual representation of the classified disease alongside the input image, offering better interpretability. Example output:

📌 **Input Image:** Leaf image uploaded by the user  
📌 **Predicted Class:** Early Blight  
📌 **Confidence Score:** 95%  
📌 **Visualization:** Heatmap overlay for affected areas  
## Inserting Value and Predicted Value

<p align="center">
  <img src="https://github.com/SinghPriya5/SinghPriya5-Potato-Disease-Classification-Using-CNN/blob/main/Potato%20Disease.png" alt="Inserting Value" width="800" height="800">
</p>

## Future Improvements
🚀 Integration with mobile applications for easy access  
🚀 Deployment on cloud platforms for scalability  
🚀 Model optimization for faster inference and reduced computational cost  
🚀 Enhancing dataset diversity for improved generalization  

## Contributors
👩‍💻 **Priya Singh**

## License
📜 This project is open-source and available under the MIT License.

Feel free to contribute and improve the project! 🚀🥔


