# Object Detection in an Urban Environment

# Importance

Object detection with AI is important for automated driving because it allows the vehicle to understand its environment and make decisions based on that understanding.
With object detection, the vehicle can identify other vehicles, pedestrians, and other objects in the environment and use that information to navigate safely and efficiently. 
This is particularly important in situations where the vehicle needs to make decisions quickly, such as when changing lanes or avoiding obstacles. 
By using AI for object detection, the vehicle can process and analyze large amounts of data in real time, enabling it to make more accurate and reliable decisions.

# Setup

I was able to use the Udacity-Workspace for this project that had all the necessary libraries and data already available. If you want to use a local setup, you can use the below instructions for a Docker container if using your own local GPU, or otherwise creating a similar environment on a cloud provider's GPU instance.

### Install Prerequisites

    pip -r requirements.txt

### Docker Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the build directory of the starter code.
The instructions below are also contained within the build directory of the starter code.
Requirements are a NVIDIA GPU with the latest driver installed and docker or nvidia-docker.

### Build the image

    docker build -t project-dev -f Dockerfile .

### Create a container :

    docker run --gpus all -v <PATH TO LOCAL PROJECT FOLDER>:/app/project/ --network=host -ti project-dev bash

### Install gsutils 
    curl https://sdk.cloud.google.com | bash

### Login to gutils
    gcloud auth login

### For further information

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation

# Data

For this project, the [Waymo Open dataset] was used (https://waymo.com/open/). Udacity provided the necessary data on their working space. 

For a local Setup the files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/).

### Download and trim subset of the data

    python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}

### Split the data into train, test and valdation

    python create_splits.py --data-dir /home/workspace/data

# Running the model

The Tf Object Detection API relies on config files. The pipeline.config is a config for a SSD Resnet 50 640x640 model.

### Download the pretrained model

    cd /home/workspace/experiments/pretrained_model/
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
    tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
    rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

### Train the model 

    python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

### Evaluate the model 

    python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config   --checkpoint_dir=experiments/reference/

### Export the trained model 

    python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/


### Create visualization

    python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif

# Experiments
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

# Exploratory Data Analysis

For an inital data exploration, 10 images were read from the dataset and plotted including the corresponding ground truth bounding boxes. The bounding boxes were color coded to differentiate between labels, red for cars, blue for pedestrians and green for cyclists. The images revealed differing weather conditions, strongly fluctuating amount of target objects and various settings. Some of the pictures and graphs are provided below. For more visualizatio got to the [exploratory data analysis notebook](https://github.com/solanhaben/UrbanObjectDetection/blob/main/Exploratory%20Data%20Analysis.ipynb).

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

In general however I would suggest a stratified k-Fold cross-validation as we are faceing a large imbalance of the target value in the dataset.
Stratified k-Fold is a variation of the standard k-Fold CV technique which is designed to be effective in such cases of target imbalance.
It works as follows. Stratified k-Fold splits the dataset on k folds such that each fold contains approximately the same percentage of samples of each target class as the complete set. 

### Training

As a first step a reference model with default hyperparameters was trained and validated. It could be observed that neither localization loss nor classification loss improved over the course of the training. 

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/ref_loss.png "References_Loss")

For an IoU treshold average of [0.5, .95] Average Precision showed 0.000 over all images and peaked at 0.002 for large area objects. For the same key metric Recall also stayed at 0.000 with a peak performance at 0.124 for large objects. The reference model overall therefore was not able to detect objects at all and needed to be improved.

<img src="https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/ref_metric.png" width="350">

# Improve the performances

## Data Augmentation Strategy

Using the https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto file containing available augmentations in the Tf Object Detection API different modifications were explored and visualized. Based on the findings in the data exploration horizonal flipping was used to combat unbalanced centering of objects. To equalize the mostly dark picture discovered in the brightness analysis, random brightness adjustments were incorporated. Lastly to mimic weather conditions that distort the picture quality patching gaussian augmentation, random adjustment of contrast and saturation were applied.

Unaugmented image             |  Augmented image
:-------------------------:|:-------------------------:
![](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/orig_img.png)  |  ![](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/augm_img.png)

## Hyperparameter Tuning

To allow model convergence, the optimizer learning rate was adjusted to 0.004, the warmup learning rate to 0.0013333 and total steps were increased to 4000. Batch size was increased from 2 to 8 to better estimate the direction of the gradient but still garanty a fast enough learning for the scope of this course.

## Result
 
Compared to the reference model the loss decreased with training steps and reached an overall lower level. This is an indicator of better performance.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/imp_loss.png "Improved_Loss")

For an IoU treshold average of [0.5, .95] Average Precision showed 0.149 over all images however the peak value at 0.753 for large area objects was a significant improvement. Average Recall also stayed quite low at 0.032 with a peak performance at 0.835 for large objects. The improved model overall was able to detect large objects to a much higher degree than the reference model but still needs to be further improved for small objects.

Precision             |  Recall
:-------------------------:|:-------------------------:
![](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/Precision_validation.png)  |  ![](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/Recall_validation.png)

<img src="https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/imp_metric.png" width="350">

This gif the shows the improved model performance on a sample set of test-images.

![alt text](https://github.com/solanhaben/UrbanObjectDetection/blob/main/pictures/animation.gif "Improved_Metric")


## Further improvements

To further increase model performance other pre-trained models could be considered. On top of that more data especially including the underrepresented labels should be used. To better detect small objects increasing image capture resolution and model input resolution is advised and additional data augmentations such as random cropping and tiling should be considered. Due to lacking ressources and limited time these steps will be taken up in future improvements.


##Acknowledgment

This project was done with Florian Binderreif as part of the Udacity Nanodegree programm. 
