# Dataset Card for Landscape Image Classification Model

## Dataset Description

- **Homepage:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data)
- **Leaderboard:** The dataset originally supported a leaderboard as part of the Intel Image Classification Challenge hosted on [DataHack by Analytics Vidhya](https://datahack.analyticsvidhya.com/).


### Dataset Summary

This dataset contains approximately 25000 images of natural scenes from various landscapes around the world. The images are 150x150 pixels in size and categorized into six different classes: buildings (0), forest (1), glacier (2), mountain (3), sea (4), and street (5). The dataset was initially created for an image classification challenge hosted by Intel on DataHack by Analytics Vidhya. The images are separated into three zip files: training (around 14000 images), testing (around 3000 images), and prediction (around 7000 images). The primary task supported by this dataset is image classification, where the goal is to classify images into one of the six scene categories.

### Supported Tasks and Leaderboards

- `image-classification`: The dataset can be used to train a model for image classification, which consists in classifying an input image into one of six categories: buildings, forest, glacier, mountain, sea, or street. Success on this task is typically measured using [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy), where a high accuracy score represents a better performance.
There is no active leaderboard, but it was originally used in the Intel Image Classification Challenge, with models ranked by accuracy.

## Dataset Structure

### Data Instances

Each instance consists of a 150x150 pixel image of a natural scene. The image data is stored in three separated zip files: `seg_train`, `seg_test`, and `seg_pred`, with images separated into six categories: buildings, forest, glacier, mountain, sea, and street. 

The dataset structure is as follows:

- **`seg_train/`**  
  - `buildings/`  
  - `forest/`  
  - `glacier/`  
  - `mountain/`  
  - `sea/`  
  - `street/`  

- **`seg_test/`**  
  - `buildings/`  
  - `forest/`  
  - `glacier/`  
  - `mountain/`  
  - `sea/`  
  - `street/`  

- **`seg_pred/`** (unlabeled images for prediction tasks)

Each image file is associated with a category label based on the folder it is stored in (for example, images in `seg_train/buildings/` are labeled as "buildings"). Here is an example format:

```
{
  'image_path': 'seg_train/buildings/example_image.jpg',
  'label': 'buildings'
}
```

### Data Fields

- **`image_path`**: The file path to the image (string).
- **`label`**: The class label of the image (string), corresponding to the folder name. The possible values are:
  - `buildings`
  - `forest`
  - `glacier`
  - `mountain`
  - `sea`
  - `street`


### Data Splits

- `Train`: 14000 images
- `Test`: 3000 images
- `Prediction`: 7000 images


## Dataset Creation

### Curation Rationale

This dataset was created to support the development of image classification models for distinguishing between different types of landscapes.

### Source Data

The images were originally published on [DataHack by Analytics Vidhya](https://datahack.analyticsvidhya.com/) for the Intel Image Classification Challenge.

#### Initial Data Collection and Normalization

The images were collected and categorized by Intel as part of the challenge. Each image was resized to 150x150 pixels for uniformity and grouped into the six categories mentioned earlier. 

### Annotations

#### Annotation process

The images were labeled based on their content (for example a forest or a glacier). These labels were likely assigned by human annotators, although there is no detailed documentation on the annotation process.

#### Who are the annotators?

The dataset's labels were likely provided by Intel or crowdworkers, but specific demographic or identity information about the annotators is not available.

### Personal and Sensitive Information

The dataset contains no personal or sensitive information, as it consists on landscape images and does not involve any identifiable human subject.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset provides a valuable resource for training and evaluating machine learning models that can classify natural scenes. Potential positive impacts include improved accuracy of scene classification algorithms, which can be used in applications like environmental monitoring and geographic information systems. However, users should be cautious of any biases in the dataset, such as an uneven distribution of landscape types that could distort the model's performance.

### Discussion of Biases

There may be biases in the dataset due to over- or under-representation of certain landscape types (for example, more images of forests or streets). This could affect the model's ability to generalize to unseen data from underrepresented categories.


## Additional Information

### Dataset Curators

The dataset was curated by Intel and hosted by DataHack (Analytics Vidhya).

### Licensing Information

There is no specific license mentioned for the dataset, but users are encouraged to respect the guidelines of the hosting platforms.

### Citation Information

```
@misc{intel_landscape_classification,
  author    = {Bansal, Puneet},
  title     = {Intel Image Classification Dataset},
  year      = {2018},
  publisher = {Kaggle},
  url = {https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data}
}
```

### Contributions

Thanks to [@puneet6060](https://github.com/puneet6060) for adding this dataset on Kaggle.
