# Violence Detection Alert System

## Overview
This project focuses on identifying violence in **real-time** from webcam footage and providing alerts. The system utilizes a combination of **computer vision** and **deep learning** models, specifically using **`VGG19`** for **spatial feature extraction** and **`LSTM`** for **temporal feature extraction**, forming the backbone of its architecture. The choice of these models enables both high accuracy of `0.95` and efficient performance, crucial for real-time violence detection.
This project has utilized **transfer learning** through a pre-trained `VGG19` model to enhance performance and reduce training time. Additionally, **multi-threading** and **asynchronous email notifications** ensure real-time detection and alerting.

  
## How It Works
*  **Capture Frames**: Video is captured in real-time. Frames are collected **resized to `160x160`** and converted to **`RGB`**.
*  **Human Detection**: The system detects the presence of humans using **`HOG descriptors`** and a **`Haar-cascade`** face detector at real time from the frames collected by the webcam and **starts recording**.
*  **Concurrent Prediction**: Once **16 frames** are collected, they are sent to the model for violence prediction. This is done concurrently using **multithreading**, allowing real-time video capture and prediction without delay.
*  **Violence detection**: A threshold of `0.6` is set to determine violence detection
*  **Asynchronous Alerts**: If 8 or more of the last 12 predictions are violent, the system triggers an asynchronous email alert to the concerned person. The email includes the recording name and the time violence was detected. Only one email is sent per recording session to avoid spamming.
*  **Recording Control**: The system stops recording if no humans are detected for 5 seconds.
  
Detecting humans again will trigger the same process


##  Model Architecture
The Violence Detection model is built using a combination of convolutional neural networks (CNN) and long short-term memory (LSTM) layers to classify video frames for violence detection. The architecture is designed to extract spatial features from individual frames and learn temporal patterns across sequences of frames, which is crucial for identifying violent actions where context and movement patterns across multiple frames are critical.
![image](https://github.com/user-attachments/assets/5cf9ac96-54ac-445f-824b-27c29bbf02d1)


##  1. Model Architecture Components
###  a. VGG19 Pre-trained Network for Spatial Feature Extraction
*  Utilizes VGG19 pre-trained on ImageNet to extract spatial features from frames.
*  Excludes top layers `include_top=False` to focus on convolutional layers.
*  **Input** is a 3-channel image of dimensions `160x160x3`, matching resized RGB frames.
*  All layers are initially frozen to retain pre-trained weights during training.
*  After passing through all convolutional layers, the output shape is `(5, 5, 512)` i.e `512` **kernels** of **size** `(5 x 5)`
###  b. Convolutional Neural Network (CNN) for Feature Extraction
*  A Sequential CNN model **wraps the VGG19 base model**.
*  A **Flatten layer** converts 2D feature maps into a 1D vector for subsequent layers
*  The output size for each frame after VGG19 is `(5 x 5 x 512) = 12800`
###  c. Time Distributed CNN for Video Processing
*  The model uses a TimeDistributed layer to **apply the CNN (VGG19) to each frame individually**, enabling feature extraction from video frame sequences.
*  Input shape for this layer is `(16, 160, 160, 3)`, where 16 is the number of frames in the sequence and 160x160x3 is the shape of each frame.
*  The TimeDistributed layer with VGG19 processes 16 frames `(160x160x3)` to produce an output shape of `(16,12800)`.
### d. LSTM for Temporal Feature Learning
*  After the CNN processes the frames, the model uses an LSTM (Long Short-Term Memory) layer to learn temporal dependencies.
*  The LSTM has 30 units and return_sequences=True to output a sequence matching the input length. outputs a shape of `(16, 30)`
*  It processes the `(16,12800)` output from the TimeDistributed layer, resulting in 16 frame representations, each of size 30.
###  e. Time Distributed Dense Layer
*  A TimeDistributed Dense layer with 90 units is applied to each frame in the sequence, refining features extracted from the CNN and LSTM layers.
*  It takes `(16,30)` input from LSTM and ouputs `(16,90)`
###  f. Global Average Pooling & Dense Layers
*  A GlobalAveragePooling1D layer averages temporal features across the sequence, reducing dimensionality while retaining essential information.
*  It reduces the input shape of `(16,90)` by averaging the 90 temporal features across all 16 sequences, resulting in a single output vector of size 90.
*  A fully connected Dense layer with `512 units` and a **`ReLU` activation** is applied, followed by **`Dropout`** to **prevent overfitting**.
### g. Final Output Layer
*  The final layer is a **Dense layer** with a single unit and a **`sigmoid activation`**, which outputs a binary classification (violence or no violence) as a probability.
##  2. Model Compilation
*  The model uses the **`Adam optimizer`** with a learning rate of **`0.0005`** for efficient gradient-based optimization.
*  **`Binary Cross-Entropy Loss`** is used as the loss function since this is a binary classification problem.
*  The model's performance is tracked using accuracy as the primary metric.


##  Performance Metrics
*  **Accuracy**: `0.953`
*  **F1 Score**: `0.952`
*  **Precision**: `0.955`
*  **Recall**: `0.95`
These metrics demonstrate the model's high efficiency in detecting violent activities.


##  Data Collection and Preprocessing
The system uses three datasets from Kaggle to train and fine-tune the model. Data was first preprocessed into smaller chunks to **hyperparameter tune** the model. Once the best parameters were identified, the model was trained on the full datasets to learn the optimal weights for violence detection.

###  Datasets:
*  [Video Fight Detection Dataset](https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset/data): 100 fight, 101 non-fight videos.
*  [Hockey Fight Videos Dataset](https://www.kaggle.com/datasets/yassershrief/hockey-fight-vidoes): 500 fight, 500 non-fight videos.
*  [Real Life Violence Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset/data): 1000 violent and 1000 non-violent videos.
###  Preprocessing:
*  **`Frame Resizing`**: Each video frame is resized to 160x160 pixels to ensure uniformity across the dataset, reducing computational complexity while retaining essential features.
*  **`BGR to RGB Conversion`**: Since the input video frames are initially in BGR format (as often used in OpenCV), they are converted to `RGB` to align with the color scheme expected by the pre-trained model VGG19.

## File Structure
*  `app.py`: Main application that handles video capture, human detection, recording, and multithreaded prediction.
*  `model.py`: Defines the model architecture using VGG19 with LSTM layers for video frame prediction.
*  `prediction.py`: Contains the prediction logic that runs concurrently and handles asynchronous tasks like sending email alerts.
*  `tasks.py`: Defines asynchronous tasks, including email alerts using Celery and Redis.

##  Technologies Used
*  **`TensorFlow`**: Deep learning framework.
*  **`Keras`**: High-level API for building neural networks.
*  **`OpenCV`**: Video capture and processing.
*  **`VGG19 + LSTM`**: Deep learning model for violence detection.
*  **`Celery`**: Asynchronous task handling for email alerts.
*  **`Redis`**: Used as a message broker for Celery.
*  **`Flask`**: Web framework for video capture and notifications.
*  **`Python`**: Programming language used for the implementation.


##  How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/DEENA0503/violence_detection_alert_system.git
2. Navigate to the project directory:
   ```bash
   cd violence_detection_alert_system
3. Start the Redis server on Ubuntu (WSL):
   ```wsl
   redis-server
4. Start the Mailhog server on Ubuntu (WSL):
   ```wsl
   Mailhog
5. Start Celery in a separate terminal:
   ```bash
   celery -A prediction worker -l info -P gevent
6. Run the Flask app:
   ```bash
   python app.py
7. Open your browser and navigate to `http://localhost:8025/` to check the alert service on Mailhog
8. 5. Press `q` to stop the webcam and exit the application.
