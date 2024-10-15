**Indian_sign_language_recognition_using_CNN**

**Project Workflow**
1) **Model Creation**

The main CNN architecture is implemented in the cnn.py file, consisting of multiple layers to perform feature extraction and object recognition.
The trained model is saved in HDF5 (Hierarchical Data Format), allowing easy access for future use.

**2) Sign Detection Execution**

Use the signdetect.py script to detect and interpret hand signs in real-time.
OpenCV is utilized to generate a bounding box, capturing hand gestures. These gestures are converted to HSV color channels (upper and lower blue) to assist with detection and identification.

**Project Overview**

Humans communicate through various languages, enabling seamless information exchange. In contrast, sign language employs visual gestures, serving as a primary mode of communication for individuals with hearing or speech impairments. Over 300 different sign languages are actively used worldwide.

The Indian Sign Language (ISL), which was standardized in 2001, is widely utilized across the country. However, communication barriers exist between individuals with hearing impairments and those without, mainly due to differences in communication modes and the lack of certified interpreters. This project aims to bridge these communication gaps by developing a system capable of detecting ISL signs, promoting effective interaction between both communities.

While research in ISL recognition is still in its early stages, this project leverages the third edition of the ISL system to create a solution that can identify alphabet signs. Using deep learning, we designed a CNN-based model that converts hand gestures into readable text.

**Methodology**

This project develops a CNN model optimized for image classification, allowing recognition of Indian Sign Language signs. The ISL dataset is preprocessed and fed into the CNN. Key elements of the model architecture include:

1.Convolutional layers for feature extraction.
2.MaxPooling layers to perform object detection.
3.A flattening layer to convert 2D feature maps into a linear vector for further processing.

By utilizing multiple layers, the CNN model ensures high accuracy and robust recognition of sign language gestures.

**Requirements**
To run the project, the following Python environment and packages are needed:

**Python packages:**
tensorflow
keras
opencv-python
Use the following commands to install the required packages:

pip install tensorflow  
pip install keras  
pip install opencv-python  

**Model Architecture**

The CNN model incorporates:

1.Convolutional and pooling layers for feature extraction and detection.
2.A flattening layer to convert 2D outputs into 1D vectors.

**Execution and Output**

Input signs are captured within a bounding box, processed into recognizable image formats, and compared against the dataset.
Predicted signs are converted into text for easy interpretation.

**Example Detections:**

Recognized signs: g, u, x
Performance Metrics:
Accuracy: 90%
Loss: 28%

**Conclusion**

This project demonstrates the ability to recognize individual letters of the Indian Sign Language (ISL) using a CNN-based deep learning model. By feeding signs into a bounding box, converting them into image data, and matching them with the dataset, the system achieves high accuracy in interpreting gestures. This work is a step forward toward building accessible communication solutions for individuals with hearing impairments.
