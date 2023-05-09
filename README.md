# CS279

This project is a white blood cell classification project that utilizes the pytorch framework to classify white blood cell images as eosinophil, lymphocyte, monocyte, or neutrophil. 

The first file, newmodel.py, is an attempt at training a machine learning model from scratch. It has some features that can be enabled, such as gaussian or high pass image filters, to possibly help with cell type classification. With a single epoch of training, the model achieved 53% accuracy.

The second file, pretrained.py, leverages the pretrained shufflenet model, and fits the model for the needs of image classification. This model, also on a single training epoch, achieved a 91% accuracy on the testing dataset.

Each of these files has a training method, which trains the given model into a .pth file, and a testing function, which useses the test dataset to assess the accuracy of a given model file. 

To run newmodel.py without high pass filter, go into MyNetwork() and comment out the high pass function. Running the code will save a .pth file on the computer which represents the model that was just trained. 
