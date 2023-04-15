# PCPNet: An Efficient and Semantic-Enhanced Transformer Network for Point Cloud Prediction

![](figs/motivation.png)
*PCPNet predicts future F range images based on the given past P sequential range images. The semantic information in the sequential range images is extracted for training, making the outputs of PCPNet closer to the ground truth in semantics.*

## Contents
1. [Publication](#Publication)
2. [Data](#Data)
3. [Installation](#Installation)
4. [Training](#Training)
5. [Testing](#Testing)
6. [Visualization](#Visualization)
7. [Download](#Dwnload)
8. [License](#License)

![](figs/overall_architecture.png)
*Overall architecture our proposed PCPNet*

## Publication
If you use our code in your academic work, please cite the corresponding [paper]():
    
```latex

```

## Data
We use the KITTI Odometry dataset for PCPNet, which you can download the dataset from the [official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).


Meanwhile, we use SemanticKITTI for semantic auxiliary training, which you can download the dataset from the [official website](http://semantic-kitti.org/dataset.html#download).

## Installation

### Source Code
Clone this repository and run 
```bash
cd PCPNet
git submodule update --init
```
to install the Chamfer distance submodule. The Chamfer distance submodule is originally taken from [here](https://github.com/chrdiller/pyTorchChamferDistance) with some modifications to use it as a submodule.

All parameters are stored in ```config/parameters.yaml```.

### Dependencies
In this project, we use CUDA 11.4, pytorch 1.8.0 and pytorch-lightning 1.5.0. All other dependencies are managed with Python Poetry and can be found in the ```poetry.lock``` file. If you want to use Python Poetry, install it with:
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```

Install Python dependencies with Python Poetry
```bash
poetry install
```

and activate the virtual environment in the shell with
```bash
poetry shell
```

## Training
We process the data in advance to speed up training. The preprocessing is automatically done if ```GENERATE_FILES``` is set to true in ```config/parameters.yaml```.

If you have not pre-processed the data yet, you need to set ```GENERATE_FILES: True``` in ```config/parameters.yaml```, and run the training script by
```bash
python train.py --rawdata /PATH/TO/RAW/KITTI/dataset/sequences --dataset /PATH/TO/PROCESSED/dataset/
```
in which ```--rawdata``` points to the directory containing the train/val/test sequences specified in the config file and  ```--dataset``` points to the directory containing the processed train/val/test sequences


If you have already pre-processed the data, you can set ```GENERATE_FILES: False``` to skip this step, and run the training script by
```bash
python train.py --dataset /PATH/TO/PROCESSED/dataset/
```
using the parameters defined in ```config/parameters.yaml```. 

To resume from a checkpoint, you run the training script by
```bash
python train.py --dataset /PATH/TO/PROCESSED/dataset/ --resume /PATH/TO/YOUR/MODEL/
```
You can also use the flag```--weights``` to initialize the weights from a pre-trained model. Pass the flag ```--help``` if you want to see more options.

A directory will be created in ```runs``` which saves everything like the model files, used config, logs and checkpoint.


## Testing
Test your model by running
```bash
python test.py --dataset /PATH/TO/PROCESSED/dataset/ --model /PATH/TO/YOUR/MODEL/
```
*Note*: Use the flag ```-s``` if you want to save the predicted point clouds for visualiztion and ```-l``` if you want to test the model on a smaller amount of data. By using the flag ```-o```, you can only save the predicted point clouds without computing loss to accelerate the speed of saving.

## Visualization
After passing the ```-s``` flag or the ```-o```flag to the testing script, the predicted range images will be saved as .png files in ```runs/MODEL_NAME/test_TIME/range_view_predictions```. The predicted point clouds are saved to ```runs/MODEL_NAME/test_TIME/point_clouds```. You can visualize the predicted point clouds by running
```bash
python visualize.py --path runs/MODEL_NAME/test_TIME/point_clouds
```

![](docs/predictions.gif)
*Five past and five future ground truth and our five predicted future range images.*

![](docs/qualitative.png)
*Last received point cloud at time T and the predicted next 5 future point clouds. Ground truth points
are shown in red and predicted points in blue.*

## Download
You can download our best performing model from [here](https://www.ipb.uni-bonn.de/html/projects/point-cloud-prediction/pretrained.zip). Just extract the zip file into ```runs```.

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
