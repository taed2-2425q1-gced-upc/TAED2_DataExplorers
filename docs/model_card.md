# Model Card for Landscape Image Classification Model

This model is designed to classify landscape images into predefined categories such as buildings, forests, glaciers, mountains, seas, and streets.

## Model Details

### Model Description

This model is based on the use of Convolutional Neural Network (CNN) architecture, specifically a Keras Sequential model, to accurately categorize various types of landscape images. It has been trained on a diverse dataset of images, enabling it to recognize different features and characteristics associated with each class.

The model was introduced on Kaggle, and the original implementation is available at [Kaggle Intel Image Classification CNN](https://www.kaggle.com/code/mohammedezzeldean/intel-image-classification-cnn). It was developed by Mohammed Ezzeldean, and any copyright issues are attributed to them.

(include general information about training procedures, parameters, and important disclaimers)

- **Developed by:** Mohammed Ezzeldean
- **Shared by:** [Kaggle](https://www.kaggle.com/)
- **Model type:** Supervised Learning, CNN
- **License:** Apache 2.0

### Model Sources

- **Repository:** [Kaggle Intel Image Classification CNN](https://www.kaggle.com/code/mohammedezzeldean/intel-image-classification-cnn)


## Uses

### Direct Use

This model can be used directly for classifying landscape images without the need for extensive fine-tuning or integration into a larger system. Users can simply input images, and the model will return the predicted class label corresponding to the landscape depicted. Some direct uses are:

- **Detection of potential anomalies:** The model can identify unusual features in landscape images, such as changes in vegetation or land erosion, which may indicate the risk of natural disasters. This capability contributes to disaster prevention efforts, helping organizations respond to mitigate loss of life and property.

- **Wildlife conservation:** Implement the model in conservation efforts to categorize and analyze images taken in natural habitats. This helps studying wildlife distribution, monitoring endangered species, and ensuring habitat preservation by understanding the ecosystems.
  

### Downstream Use

The model can also be integrated into larger applications or systems for more complex tasks. For instance, it can be fine-tuned on specific subsets of data or incorporated into an automated pipeline for processing large datasets of landscape images. Some potential downstream uses include:

- **Fine-tuning for specific domains:** The model can be adapted to classify specialized subsets of landscape images, such as agricultural fields, urban areas, or protected natural reserves. Fine-tuning allows users to enhance accuracy for specific applications, ensuring that the model meets the correct needs.

- **Automated image processing pipelines:** By integrating the model into automated workflows, users can ease the process of analyzing large amounts of landscape images. This is particularly beneficial for organizations involved in environmental monitoring, or large-scale agricultural assessments, enabling quicker decision-making based on real-time data.

### Out-of-Scope Use

While the landscape classification model offers a range of applications, there are specific uses for which it may not be suitable or effective. Understanding these limitations helps users manage expectations and identify appropriate alternatives. Some out-of-scope uses include:

- **High-resolution detail analysis:** The model may not perform well in tasks requiring the analysis of details within landscapes, such as identifying individual plants or animals. For such applications, more specialized models focusing on high-resolution images or object detection might be necessary.

- **Integration with non-visual data:** The model focuses solely on image classification and does not handle non-visual data types, such as textual information or numerical datasets. Integrating it into systems requiring other types of data would need additional models or tools capable of processing those data types.


## Bias, Risks, and Limitations

The model's performance is influenced by several factors, leading to potential biases and limitations:

- **Dataset bias**: The training dataset may not cover the full diversity of landscape types, leading to underperformance on classes that are underrepresented. 

- **Image quality**: The classification accuracy can be significantly impacted by the quality of the input images. High-resolution images with clear details tend to yield better predictions, while low-resolution images or those with compression artifacts may lead to misclassification.

- **Environmental conditions**: Variability in environmental conditions at the time the images were captured can also affect performance. Factors such as the weather (cloud cover, rain, or snow) can hide important visual features in landscape images, as well as seasonality, which alter the appearance of these scenes, potentially leading to confusion in classification.

- **Limitations in generalization**: The model may struggle to generalize effectively to new landscape types that differ from those seen during training, which can limit its applicability in real-world scenarios.

### Recommendations

To mitigate these biases and limitations, it is advisable to:

- **Diversify training data**: Include a wider variety of landscape types and conditions in the training dataset to improve the model's robustness.

- **Enhance image quality**: Use high-resolution images and consider pre-processing techniques to enhance quality and reduce artifacts.

- **Consider environmental context**: Be aware of the conditions under which images are taken and evaluate how these factors may influence predictions.


## How to Get Started with the Model

Use the code below to get started with the model.

```python

import numpy as np
import cv2
import pickle

# Load the model
with open('path_to_the_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the image
image = cv2.imread('path/to/your/image.jpg')
image = cv2.resize(image, (100, 100)) / 255.0   # Resize and normalize
image = np.expand_dims(image, axis=0)           # Expand dimensions

# Make predictions
predictions = model.predict(image)
predicted_class = np.argmax(predictions)

print(predicted_class)

```

## Training Details

### Training Data

The model was trained on the Intel Image Classification dataset, which contains over 14000 landscape images categorized into six classes: buildings, forests, glaciers, mountains, seas, and streets. For more details, please refer to the [dataset card](https://github.com/taed2-2425q1-gced-upc/TAED2_DataExplorers/blob/main/docs/dataset_card.md).

### Training Procedure


#### Preprocessing: 
The preprocessing workflow establishes paths for training, testing, and prediction datasets, and maps each landscape category to a numeric code. It resizes all images to 100x100 pixels, collects images and labels into lists, and converts these lists into Numpy arrays, which are then saved in `.npy` format for efficient future use.

#### Training overview:
The `train.py` script trains a Convolutional Neural Network (CNN) for image classification using MLflow for experiment tracking and logging. It loads preprocessed training data from '.npy' files, constructs and compiles the CNN, and tracks carbon emissions during training. After training, the model is saved as a pickle file for future use.

#### Training Hyperparameters

- `Batch Size`: 64
- `Epochs`: 20
- `Optimizer`: Adam with learning rate 0.0001
- `Loss Function`: Sparse Categorical Crossentropy
- `Metrics`: Accuracy

#### Additional notes:
- **Emissions Tracking**: The emissions tracker ensures the environmental impact of the model training process is monitored and logged, promoting awareness of carbon footprints in machine learning practices.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated using a validation dataset consisting of landscape images that were not part of the training set. The dataset includes images from all six categories: buildings, forests, glaciers, mountains, seas, and streets

#### Factors

Key factors influencing the evaluation results include the quality of the input images, variations in lighting and environmental conditions, and the diversity of landscapes represented in the validation dataset. Additionally, not all categories have the same number of images, which can impact the balance of the dataset. The inherent biases in the training data can also affect how well the model generalizes to unseen data.

#### Metrics

The model's performance is measured using accuracy as the primary metric, which reflects the proportion of correctly classified images. Accuracy is particularly useful in this context because it provides a straightforward measure of how often the model makes correct predictions across all classes. Other metrics such as mean absolute error (MAE) and mean squared error (MSE) may also be utilized to provide a comprehensive assessment of the model's predictive capabilities.

### Results

The evaluation resulted in an accuracy score of [Accuracy Value Here]. This score indicates how well the model classifies landscape images into the correct categories. While accuracy gives a good overview of performance, it might not show all the model's strengths and weaknesses, especially if the dataset is imbalanced or if certain classes are more challenging to classify than others. Future evaluations could include more metrics, like a confusion matrix, to better understand misclassifications and identify fututre improvements of the model.

#### Summary

Overall, the model demonstrates a good performance, effectively classifying landscape images across most categories. However, there may be limitations in accurately identifying certain categories, particularly those with similar features or lower representation in the dataset. Enhancing dataset diversity and refining the model architecture could lead to better classification results in future iterations.

## Environmental Impact

The carbon emissions of all the training process and experiments made estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** T4x2
- **Hours used:** 6
- **Cloud Provider:** Kaggle
- **Carbon Efficiency (kg/kWh):** 0.432
- **Offset bought (%)**: 0
- **Carbon Emitted (kg CO2):** 0.4

## Technical Specifications

### Model Architecture and Objective

The model uses a CNN architecture designed to classify six landscape categories, including buildings, forests, glaciers, mountains, seas, and streets. It consists of multiple convolutional layers that extract spatial hierarchies of features, followed by max-pooling layers to reduce dimensionality and improve computational efficiency. Dense layers are included towards the end of the architecture, culminating in a softmax output layer that enables multi-class classification by providing probability distributions over the six classes.

### Compute Infrastructure

#### Hardware

- **GPU**: Can be trained using any GPU platform (e.g., NVIDIA Tesla P100, V100, or higher)
- **RAM**: 16GB or more recommended for faster training
- **Storage**: Sufficient local or cloud storage required to handle the dataset and model files

#### Software

- **Keras** (TensorFlow backend): For deep learning model building
- **Python**
- **OpenCV**: For image processing

**BibTeX:**

```
@misc{Intel_Image_Classification_CNN,
  author = {Ezzeldean, Mohammed},
  title = {Intel Image Classification CNN},
  year = {2024},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/code/mohammedezzeldean/intel-image-classification-cnn}
}
```

**APA:**

Ezzeldean, M. (2024). *Intel Image Classification CNN*. Kaggle. https://www.kaggle.com/code/mohammedezzeldean/intel-image-classification-cnn

## Glossary

- **Convolutional Neural Network (CNN)**: A class of deep neural networks commonly used in image recognition and processing tasks, which uses a mathematical operation called convolution to extract features from images.
- **Epoch**: One complete cycle through the entire training dataset during the training process of a machine learning model.
- **Batch Size**: The number of training samples utilized in one iteration of the model's learning process.
- **Sparse Categorical Crossentropy**: A loss function used in multi-class classification problems that measures the performance of a model whose output is a probability value between 0 and 1.
- **MLflow**: An open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.


## Model Card Authors

Martina Albà González

Sara Alonso del Hoyo

Marc Colomer Xaus

Carlota Gozalbo Barriga

## Model Card Contact

martina.alba@estudiantat.upc.edu

sara.alonso.del.hoyo@estudiantat.upc.edu

marc.colomer.xaus@estudiantat.upc.edu

carlota.gozalbo@estudiantat.upc.edu
