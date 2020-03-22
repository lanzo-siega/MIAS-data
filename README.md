# MIAS-data

## Files:
* __setup.py__: code reading the csv file and creating the training and test data sets
* __mias_cnn.py__: code for creating the Convolutional Neural Network model
* __acc_graph.jpg__: graph of the model's accuracy per epoch

## Dataset:
__Link to dataset__: http://peipa.essex.ac.uk/info/mias.html

## Methodology and Results:
  The first step of this code is to load images from the directory, convert them to arrays, then append all the arrays to a list. This step translates the images for the CNN model. After the list is created, it must be split into training and validation sets. This allows the model to evaluate its accuracy and adjust accordingly. After the training and validation sets are created, it is then fed through the model. The model uses a Convolutional Neural Network (CNN) as the framework with three convolutional layers, two dense layers, and one flattened layer. While the list of translated images is being passed through the model, the accuracy for each epoch is also being graphed. 
