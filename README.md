## Project Overview

This project aims to classify images of animals into ten categories using a CNN model. The model utilizes transfer learning by fine-tuning a pre-trained ResNet50 architecture, which helps in achieving better performance with fewer training data.

## Dataset

The dataset used in this project is the [Animals-10 dataset on Kaggle]([https://www.robots.ox.ac.uk/~vgg/data/animals/](https://www.kaggle.com/datasets/alessiocorrado99/animals10)). It contains images of 10 different animal classes:
- Cats
- Dogs
- Horse
- Elephants
- Butterfly
- Chicken
- Cow
- Sheep
- Squirrel
- Spider


## Installation

To set up the project, you will need to create a virtual Conda environment and install the necessary dependencies. The environment can be set up using the provided `environment.yml` file.

1. **Clone the repository**:
```
git clone https://github.com/musheenlernin/Animals-10CNN.git
```
3. **Navigate into the project directory**
```
cd Animals-10CNN
```
4. **Create and activate the conda environment**:
```
conda env create -f environment.yml
conda activate animals-10cnn
```

## Usage
To train the model, run the following command:
```python train.py```
After training, the model will save the best weights in a file best_model.pth

## Visualizations

The project includes various visualizations to help understand model performance:

* Training and validation loss and accuracy curves.
* Confusion matrix to visualize prediction results.

These visualizations provide insights into the model's performance and areas for improvement.

