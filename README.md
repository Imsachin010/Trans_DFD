# Deepfake Detection Solution

## Proposed Solution
We propose an effective Deepfake detection system by leveraging Deep Learning models at a mesoscopic level of analysis. Our approach uses a deep neural network architecture with a minimal number of layers, specifically designed to identify falsified videos.

### Key Features:
- **Dual Forgery Detection**: Successfully detects two types of video tampering—Deepfake and Face2Face—forgeries.
- **Transfer Learning**: The solution utilizes Transfer Learning by implementing the MesoNet-4 architecture, trained on a custom dataset to enhance its accuracy in detecting deepfakes.
  
### Model Architecture:
- **MesoNet-4**: 
  - The convolutional layers use **ReLU** activation functions, introducing non-linearities.
  - **Batch Normalization** is applied to regularize the output of the convolutional layers.
  - The fully-connected layers employ **Dropout** to prevent overfitting.

### Deployment:
The trained model (`Mesonet.h5`) is deployed as a web application using **Streamlit**. Users can upload video files to analyze and receive a Deepfake detection result in real-time.

### Repository:
- **GitHub Repository**: [deepfake-guard-AI](https://github.com/Imsachin010/Trans_DFD/)
- **AI/ML Approach**: Deep learning model built in Python.
- **User Interface**: The web app features a simple and intuitive UI for video upload and result display.

## Results:
For more details on the detection results, architecture, and the user experience, [click here](#).

