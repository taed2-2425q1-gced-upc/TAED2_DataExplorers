---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

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
- **Language(s) (NLP):** Not Applicable
- **License:** 
- **Finetuned from model [optional]:**

### Model Sources [optional]

- **Repository:** [Kaggle Intel Image Classification CNN](https://www.kaggle.com/code/mohammedezzeldean/intel-image-classification-cnn)


## Uses

### Direct Use

This model can be used directly for classifying landscape images without the need for extensive fine-tuning or integration into a larger system. Users can simply input images, and the model will return the predicted class label corresponding to the landscape depicted. Below is a simple example of how to use the model:

```python

import numpy as np
import cv2
from keras.models import load_model

# Load the model
model = load_model('path_to_your_model.h5')

# Prepare your image
image = cv2.imread('path/to/your/image.jpg')

# Make predictions
predictions = model.predict(image)

greet()
```

### Downstream Use [optional]

The model can also be integrated into larger applications or systems for more complex tasks. For instance, it can be fine-tuned on specific subsets of data or incorporated into an automated pipeline for processing large datasets of landscape images.

### Out-of-Scope Use

While the model is designed for landscape image classification, certain uses are not recommended due to potential inaccuracies or ethical concerns. Using the model to classify images that do not contain landscapes can lead to poor performance and misleading results. Low-quality or distorted images may also produce unreliable predictions, therefore users should ensure image quality before processing. Finally, avoid using the model in contexts where it may reinforce stereotypes or biases about certain landscapes, which could lead to unintended negative consequences.

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


## Training Details

### Training Data

The model was trained on the Intel Image Classification dataset, which contains over 14000 landscape images categorized into six classes: buildings, forests, glaciers, mountains, seas, and streets. For more details, please refer to the [dataset card](https://github.com/taed2-2425q1-gced-upc/TAED2_DataExplorers/blob/main/docs/dataset_card.md).

### Training Procedure


#### Preprocessing [optional]



#### Training Hyperparameters



#### Speeds, Sizes, Times [optional]


## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware_type | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

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
