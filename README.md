# Object Detection in an Urban Environment

## Data

For this project, the [Waymo Open dataset] was used (https://waymo.com/open/).

The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. For this project the data was already provided within a udacity workingspace

## Structure

The data used for training, validation and testing is organized as follow:
```
data/
    - train: contains 86 files for model training
    - val: contains 10 files for model validation
    - test: contains 3 files to test the model and create inference videos
```

### Experiments
The experiments folder is organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference model with the unchanged config file
    - augmentation/ - containing the model config including data augmentation and hyperparameter tuning
    - label_map.pbtxt
```

## Prerequisites

pip -r requirements.txt

## Instructions

### Exploratory Data Analysis

For an inital data exploration, 10 images were read from the dataset and plotted including the corresponding ground truth bounding boxes. The bounding boxes were color coded to differentiate between labels, red for cars, blue for pedestrians and green for cyclists. The images revealed differing weather conditions, strongly fluctuating amount of target objects and various settings. Some of the pictures and graphs are provided below. For more visualizatio got to the exploratory data analysis notebook (https://github.com/solanhaben/UrbanObjectDetection/blob/main/Exploratory%20Data%20Analysis.ipynb).

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/visual.png "Example Picture")

For further exploration, 20.000 datapoints for taken from the dataset and statistically analysed:

Plotting the label distribution across the dataset revealed a heavily unbalanced distribution. Class 1 ('cars) made up 78.6% of objects, followed by class 2 ('pedestrians') with 20.9%. Class 4 ('cyclists) was only represented with 0.5%.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/label_dis.png "Lable_Dist")

To explore the bounding box positions within the dataset, the center-pixel for each object was calculated. The resulting values were transfered to a heatmap, grouping and adding pixels within a 20x20 array. The heatmap shows areas with a higher density of object-centers in brigther colours and vice versa darker colours in areas with a lower accumulation of object centers. Class 1 and Class 4 showed a slight tendency to be located on the left side of the picture and class 2 a higher density towards the right border. 

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/bbox_pos.png "Class1_bbox_dist")

Furthermore the average lightness within the dataset was investigated. For this step, the RGB images were converted to HSV images and reshaped. Next the mean of the brightness across all pixel was calculated and saved. To display the distribution of lightness a histogramm on the entire dataset was created. The distribution is bimodal with peaks at 0.3 and 0.4 but fairly symetrical.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/lightness.png "Lightness")

Lastly the label count for each each picture was calculated and plotted on a frequency histogramm. Class 1 was represented between 0 and 68 times with 7 being the highest frequency. Most pictures contained less than 40 cars. Class 2 ranged from 0 to 43 with most images have no pedestrian. class 4 ranged from 0 to 6 representations but almost 90% of images do not contain a 'cyclist'.

### Cross Validation

    Inside of the Udacity Workspace the train/test/validation split was already provided and splitted the dataset into:
    - train: containing 86 files
    - val: containing 10 files
    - test - containing 3 files


### Training

As a first step a reference model with default hyperparameters was trained and validated. It could be observed that neither localization loss nor classification loss improved over the course of the training. 

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/ref_loss.png "References_Loss")

For an IoU treshold average of [0.5, .95] Average Precision showed 0.000 over all images and peaked at 0.002 for large area objects. For the same key metric Recall also stayed at 0.000 with a peak performance at 0.124 for large objects. The reference model overall therefore was not able to detect objects at all and needed to be improved.

<img src="https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/ref_metric.png" width="350">

### Improve the performances

# Data Augmentation Strategy

Using the https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto file containing available augmentations in the Tf Object Detection API different modifications were explored and visualized. Based on the findings in the data exploration horizonal flipping was used to combat unbalanced centering of objects. To equalize the mostly dark picture discovered in the brightness analysis, random brightness adjustments were incorporated. Lastly to mimic weather conditions that distort the picture quality patching gaussian augmentation, random adjustment of contrast and saturation were applied.

# Hyperparameter Tuning

To allow model convergence, the optimizer learning rate was adjusted to 0.004, the warmup learning rate to 0.0013333 and total steps were increased to 4000. Batch size was increased from 2 to 8 to better estimate the direction of the gradient but still garanty a fast enough learning for the scope of this course.

# Result
 
Compared to the reference model the loss decreased with training steps and reached an overall lower level. This is an indicator of better performance.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/imp_loss.png "Improved_Loss")


For an IoU treshold average of [0.5, .95] Average Precision showed 0.149 over all images however the peak value at 0.753 for large area objects was a significant improvement. Average Recall also stayed quite low at 0.032 with a peak performance at 0.835 for large objects. The improved model overall was able to detect large objects to a much higher degree than the reference model but still needs to be further improved for small objects.

<img src="https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/imp_metric.png" width="350">

This gif the shows the improved model performance on a sample set of test-images.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/animation.gif "Improved_Metric")


# Further improvements

To further increase model performance other pre-trained models could be considered. On top of that more data especially including the underrepresented labels should be used. To better detect small objects increasing image capture resolution and model input resolution is advised and additional data augmentations such as random cropping and tiling should be considered. Due to lacking ressources and limited time these steps will be taken up in future improvements.
