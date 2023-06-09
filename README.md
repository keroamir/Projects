# Skin Cancer Detector using Transfer Learning with CNN and PyTorch

🔬 Skin cancer detector that leverages transfer learning with Convolutional Neural Networks (CNN) and PyTorch to classify images into benign (non-cancerous) and malignant (cancerous) categories.

## Project Overview

📚 This project focuses on developing a skin cancer detector using transfer learning with CNN and PyTorch. The detector aims to assist in the early detection and diagnosis of skin cancer, improving patient outcomes.

## Steps Taken in the Project

1. **Data Loading**: The first step involves loading the skin cancer dataset. The dataset is organized into separate folders for training, validation, and testing.

2. **Data Preprocessing**: The dataset undergoes preprocessing, including transformations such as random rotation, resizing, cropping, horizontal flipping, and normalization. Each data subset (train, validation, test) has its specific transformations.

3. **Data Loaders**: Data loaders are created using PyTorch's `DataLoader` class, providing an efficient way to load and iterate over the dataset during training and testing.

4. **Visualization**: A batch of training data is visualized using matplotlib, offering a glimpse of the images and their corresponding labels.

5. **Loss Function and Optimizer**: The loss function (Negative Log-Likelihood Loss) and optimizer (Adam) are defined for the model's training.

6. **Model Architecture**: The model architecture is based on transfer learning with the ResNet-50 CNN architecture. Pre-trained weights from the ImageNet dataset are loaded, and a custom classifier is added.

7. **Training Algorithm**: The training algorithm iterates over the specified number of epochs, optimizing the model's parameters through backpropagation and gradient descent. Training and validation losses are computed and printed for each epoch.

8. **Testing Algorithm**: The testing algorithm evaluates the model's performance on the testing set, calculating the loss and accuracy.

9. **Skin Cancer Detection Algorithm**: The final step implements the skin cancer detection algorithm. Given an image path, the algorithm applies the necessary transformations, feeds the image through the trained model, and outputs the predicted class label.

## Usage

To use this skin cancer detector, follow these steps:

1. 📦 Install the required dependencies mentioned in the Installation section of the README.

2. 🔗 Clone the repository and navigate to the project directory.

3. 📂 Prepare your skin cancer dataset and organize it into separate folders for training, validation, and testing. You can download the dataset using the following links:

   - [Training Data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip)
   - [Validation Data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip)
   - [Test Data](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip)

4. ▶️ Run the training script to train the model. Adjust the number of epochs and other parameters as needed.

5. ⚖️ Run the testing script to evaluate the model's performance on the testing set. Provide the paths to the dataset directory and the saved model file.

6. 🕵️‍♀️ To detect skin cancer in a specific image, use the `Skin_Disease_Detector` function. Provide the path to the image, and the function will display the image and output the predicted skin disease class.

## Contributing

Contributions to this skin cancer detector project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.


## Acknowledgments

The skin cancer dataset used in this project is provided by Udacity's Deep Learning Nanodegree program.

## References

- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- torchvision: [https://pytorch.org/vision/](https://pytorch.org/vision/)
