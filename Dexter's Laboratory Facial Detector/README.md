# Dexter's Laboratory 🤖

**Dexter's Laboratory** is a computer vision project focused on facial detection of characters from the animated series.

## Features 🔍
- **Facial Detection and Recognition**: The system uses **Histograms of Oriented Gradients (HOG)** for feature extraction and **Linear Support Vector Machines (SVM)** for facial classification. 🧑‍🦱
- **Data Augmentation**: Utilized the **Albumentations** library for image transformations like flipping, scaling, rotation, and perspective shifts to increase dataset variability. 🔄
- **Preprocessing**: Includes steps like histogram equalization, Gaussian blur, and HOG descriptor generation to improve face detection. 🎨
- **Training**: The model is trained on positive and negative examples of character faces. 📚
- **Evaluation**: Performance is assessed using precision-recall curves and metrics like **Average Precision (AP)**. 📊

## Technologies Used 💻
- **HOG (Histograms of Oriented Gradients)** 🖼️
- **SVM (Support Vector Machine)** 🧠
- **Albumentations Library** 🔧
- **Image Preprocessing Techniques** 🛠️

