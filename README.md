# Deepfake Detection for Tamil Speech

## Dataset Links

- Real: https://drive.google.com/drive/folders/1bK68y7nNmjpwaIdFfQOWIIwXJPTVUYlr?usp=sharing
- Fake: https://drive.google.com/drive/folders/1XUKT7kroxYVCWwu1B7Zr7uowLfMPPi9y?usp=sharing

## Abstract

In today's digital landscape, the rise of deepfake content poses a serious challenge to maintaining trust and authenticity online. This paper focuses on detecting deepfake Tamil speech by leveraging advanced machine-learning techniques. Our dataset consists of two classes: "fake" and "real," representing synthetic and authentic Tamil speech recordings, respectively. We use simple yet effective methods like Chromagram, MFCC, and Mel spectrogram to extract key features from the audio data. These features are then fed into two models: a Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM) and a Support Vector Machine (SVM). The CNN-LSTM excels at identifying patterns in spectrogram data, while the SVM handles classification tasks efficiently. Our experiments demonstrate the effectiveness of our approach in accurately distinguishing between genuine and falsified Tamil speech. This research contributes to enhancing the security and credibility of digital content in Tamil-speaking communities. By employing straightforward machine learning techniques, we aim to combat the spread of synthetic media manipulation and foster a safer online environment for all users.

## 1. Introduction

In recent years, the proliferation of synthetic media, particularly deepfake audio, has ignited significant concerns surrounding the authenticity and reliability of digital content. Among the many languages targeted by deepfake synthesis techniques, Tamil, a prominent Dravidian language predominantly spoken in South India and Sri Lanka, has emerged as a prime focal point. Its widespread usage across diverse domains, including entertainment, politics, and social media, renders it susceptible to the dissemination of manipulated content. The emergence of deepfake technology poses a serious challenge to digital trust, online security, and the integrity of information shared on digital platforms. As deepfake techniques continue to advance, it becomes increasingly difficult for individuals and automated systems to discern between genuine and manipulated audio recordings accurately.

To address this pressing issue, this research aims to develop effective deepfake detection models tailored specifically for Tamil speech. By leveraging advanced machine learning techniques such as Chromagram, MFCC (Mel-frequency cepstral coefficients), and Mel Spectrogram for audio feature extraction, coupled with CNN-LSTM (Convolutional Neural Network-Long Short-Term Memory) and SVM (Support Vector Machine) models for classification, we strive to achieve high accuracy in distinguishing between authentic and synthesized Tamil speech.

The primary objective of this project is to develop robust and effective deepfake detection models specifically tailored for the Tamil language. The aim is to create systems capable of accurately distinguishing between genuine and synthesized Tamil speech, thereby enhancing the security and credibility of digital content within Tamil-speaking communities. To achieve this objective, the research will focus on leveraging advanced machine learning techniques, including Chromagram, MFCC (Mel-frequency cepstral coefficients), and Mel Spectrogram for audio feature extraction. These techniques are chosen for their effectiveness in capturing relevant audio characteristics and patterns essential for differentiating between authentic and manipulated speech. Furthermore, the project seeks to integrate deep learning architectures, such as CNN-LSTM (Convolutional Neural Network-Long Short-Term Memory) and SVM (Support Vector Machine) models, for classification purposes. By harnessing the capabilities of these models, the goal is to achieve high accuracy in detecting deepfake audio, thus mitigating the spread of manipulated content and safeguarding the integrity of digital information.

## 2. Model Architectures

### 2.1 CNN-LSTM

#### 2.1.1 Architecture

The architecture of the CNN-LSTM model is designed to effectively capture temporal and spatial features from the input audio data. It begins with two Conv1D layers, each followed by a MaxPooling1D layer to extract high-level features and reduce the spatial dimensions of the data. The subsequent LSTM layers allow the model to learn temporal dependencies and patterns within the audio sequences. The first LSTM layer returns sequences to the second LSTM layer, which aggregates the learned features. A Dense layer with ReLU activation is employed to further process the extracted features, followed by a Dropout layer to prevent overfitting. Finally, a Dense layer with a sigmoid activation function produces the binary classification output.

#### 2.1.2 Parameters

The CNN-LSTM model is configured with specific parameters to optimize its performance. It utilizes 32 and 64 filters in the Conv1D layers with a kernel size of 3 to capture spatial features effectively. MaxPooling1D layers with a pool size of 2 are employed to downsample the feature maps. The LSTM layers consist of 64 units each to capture temporal dependencies in the audio sequences. A dropout rate of 0.5 is applied in the Dropout layer to prevent the model from relying too heavily on specific features during training.

#### 2.1.3 Evaluation Metrics

During model compilation, the Adam optimizer is chosen for optimization, which dynamically adjusts the learning rate during training. Binary cross-entropy loss function is selected as it is well suited for binary classification tasks. The model's performance is evaluated using accuracy as the metric, which measures the proportion of correctly classified instances out of the total instances. These evaluation metrics provide insights into the model's effectiveness in distinguishing between real and deepfake audio recordings.

### 2.2 SVC

#### 2.2.1 Architecture

For the Support Vector Machine (SVM) model, the architecture is inherently different from neural network-based models like CNN-LSTM. SVM operates on the principle of finding the hyperplane that best separates the different classes in the feature space. In this implementation, a linear kernel is utilized, meaning the decision boundary is a linear function of the input features.

#### 2.2.2 Parameters

The primary parameter specified for the SVM model is the kernel type, which is set to 'linear' to create a linear decision boundary.

#### 2.2.3 Evaluation Metrics

To evaluate the SVM model's performance, accuracy is employed as the metric, which measures the proportion of correctly classified instances in the test set. Accuracy score is calculated by comparing the predicted labels with the true labels of the test data. This metric provides an indication of how well the SVM model can discriminate between real and deepfake audio recordings based on the extracted features.

### 2.3 CNN-GRU

#### 2.3.1 Architecture

The architecture of the CNN-GRU model comprises convolutional layers followed by GRU (Gated Recurrent Unit) layers. The convolutional layers aim to extract spatial features from the input data, while the GRU layers capture temporal dependencies in the sequential data. Each convolutional layer is followed by Leaky ReLU activation and MaxPooling1D to downsample the data. The GRU layers are stacked to further capture complex temporal patterns in the data. Finally, a Flatten layer is used to flatten the output before passing it through a Dense layer with a sigmoid activation function to generate the final output.

#### 2.3.2 Parameters

The model parameters include the number of filters, kernel size, and strides for each convolutional layer. In this implementation, the CNN-GRU model consists of three convolutional layers with 64, 128, and 256 filters, respectively. The kernel size is set to 5, and the strides are set to 1. The GRU layers have 64, 128, and 256 units, respectively. Additionally, the model utilizes the Adam optimizer with a learning rate of 0.0001 and binary cross-entropy loss function for optimization.

#### 2.3.3 Evaluation Metrics

For evaluating the CNN-GRU model, the validation loss and accuracy are used as evaluation metrics. The model's performance is monitored using the validation loss, and early stopping is implemented to prevent overfitting by restoring the best weights based on the validation loss. The accuracy metric provides insights into the model's classification performance on the validation dataset.

## 3. Evaluation Metrics

### 3.1 Validation Accuracy

Validation Accuracy quantifies the proportion of correctly classified instances in the validation dataset, indicating the overall performance of the model in terms of correct predictions.

#### FIG 4: ACCURACY

### 3.2 Confusion Matrix

This matrix provides a detailed breakdown of model performance by classifying predictions into four categories: true positives, true negatives, false positives, and false negatives. It enables us to assess the model's ability to correctly classify instances of each class and identify any misclassifications. By analyzing the confusion matrix, we gain valuable insights into the model's strengths and weaknesses across different classes, facilitating informed decisions for model refinement and improvement.

#### FIG 5: CONFUSION MATRIX

## 4. Tools Used

### 4.1 Librosa

A Python library for analyzing and extracting features from audio signals.

### 4.2 NumPy

A fundamental package for scientific computing with Python, used for numerical operations and array manipulation.

### 4.3 Scikit-learn

A machine learning library in Python, providing simple and efficient tools for data mining and data analysis. It includes various algorithms for classification, regression, clustering, and more.

### 4.4 TensorFlow / Keras

TensorFlow is an open-source machine learning framework developed by Google, while Keras is a high-level neural networks API running on top of TensorFlow. Both are utilized for building, training, and evaluating deep learning models.

### 4.5 Convolutional Neural Networks (CNNs)

Deep learning models commonly used for image processing tasks, such as feature extraction and image classification. In this project, CNNs are adapted for processing one-dimensional audio data.

### 4.6 Long Short-Term Memory (LSTM)

A type of recurrent neural network (RNN) architecture capable of learning long-term dependencies in sequential data. LSTMs are employed for capturing temporal patterns in the audio data.

### 4.7 Gated Recurrent Unit (GRU)

Another type of recurrent neural network similar to LSTM but with simplified gating mechanisms. GRUs are utilized for capturing temporal dependencies in sequential data, including audio signals.

### 4.8 Support Vector Machine (SVM)

A supervised learning algorithm used for classification tasks. SVMs are utilized as a traditional machine learning approach for comparison with deep learning models.

### 4.9 Mel-frequency Cepstral Coefficients (MFCC)

A feature extraction technique widely used in speech and audio processing. MFCCs capture the spectral characteristics of audio signals.

### 4.10 Chroma STFT

Another feature extraction technique that computes the Short-Time Fourier Transform (STFT) of audio signals, focusing on the pitch content.

### 4.11 Data Augmentation

Techniques such as stretching, pitch shifting, adding noise, and shifting audio segments are applied to increase the diversity of the training data and improve model generalization.

### 4.12 Confusion Matrix

A performance measurement tool used to evaluate the effectiveness of a classification model, providing insights into the model's predictive accuracy and errors.

## 5. Techniques Used

### 5.1 Data Augmentation

Techniques such as stretching, pitch shifting, adding noise, and shifting audio segments are applied to increase the diversity of the training data and improve model generalization.

### 5.2 Feature Extraction

- **Mel-frequency Cepstral Coefficients (MFCC):** Captures the spectral characteristics of audio signals, commonly used in speech and audio processing.
- **Chroma STFT:** Computes the Short-Time Fourier Transform (STFT) of audio signals, focusing on the pitch content.

### 5.3 Model Architectures

- **Convolutional Neural Networks (CNNs):** Adapted for processing one-dimensional audio data, used for feature extraction and classification.
- **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) architecture capable of learning long-term dependencies in sequential data, is employed for capturing temporal patterns in audio data.
- **Gated Recurrent Unit (GRU):** Similar to LSTM but with simplified gating mechanisms, utilized for capturing temporal dependencies in sequential data, including audio signals.

### 5.4 Machine Learning Algorithm

- **Support Vector Machine (SVM):** A supervised learning algorithm used for classification tasks is employed as a traditional machine learning approach for comparison with deep learning models.

### 5.5 Evaluation Metrics

- **Validation Loss:** A metric used to evaluate the performance of the model during training and validation.
- **Validation Accuracy:** Measures the accuracy of the model's predictions on the validation dataset, providing insights into its overall performance.

### 5.6 Confusion Matrix

A performance measurement tool used to evaluate the effectiveness of a classification model, providing insights into the model's predictive accuracy and errors.


