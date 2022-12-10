import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

from scipy import ndimage

from torchvision import models
from torchvision.models import shufflenet_v2_x1_0
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models, utils
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import hickle as hkl

import glob
from tqdm import tqdm

from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Loading and normalizing the data.
# Define transformations for the training and test sets

train_transforms = A.Compose([
    A.SmallestMaxSize(max_size=350),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.SmallestMaxSize(max_size=350),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_data_path = 'dataset2-master\dataset2-master\images\TRAIN'
test_data_path = 'dataset2-master\dataset2-master\images\TEST'

train_image_paths = []  #to store image paths in list
classes = []  #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '\*'):
    train_image_paths.append(glob.glob(data_path + '\*'))

train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])

#2.
# split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(
    0.8 *
    len(train_image_paths))], train_image_paths[int(0.8 *
                                                    len(train_image_paths)):]

#3.
# create the test_image_paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '\*'):
    test_image_paths.append(glob.glob(data_path + '\*'))

test_image_paths = list(flatten(test_image_paths))

batch_size = 1
number_of_labels = 1

classes = ('EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL')

idx_to_class = {i: j for i, j in enumerate(classes)}
class_to_idx = {value: key for key, value in idx_to_class.items()}


class MySet(Dataset):

    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


train_dataset = MySet(train_image_paths, train_transforms)
valid_dataset = MySet(valid_image_paths, test_transforms)
test_dataset = MySet(test_image_paths, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.pretrained_model = shufflenet_v2_x1_0(pretrained=True)
        print(self.pretrained_model.fc.weight.shape)
        # self.pretrained_model.fc = nn.Linear(1024, 4)
        self.pretrained_model.fc = nn.Identity(1024, 1024)
        self.fc = nn.Linear(1024, 4)

    def forward(self, input):
        self.pretrained_model.eval()
        output = self.pretrained_model.forward(input)
        output = self.fc(output)
        return output


# Instantiate a neural network model
model = Network()

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def saveModel():
    path = "./HPModel.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():

    model.eval()
    accuracy = 0.0
    total = 0.0

    count = 0

    with torch.no_grad():
        for data in test_loader:
            if count > 100:
                break
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            print(outputs.shape)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            count += 1

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return (accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 100 == 99:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print(
            'For epoch', epoch + 1,
            'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(utils.make_grid(images))

    # Show the real labels on the screen
    print('Real labels: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ',
          ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))


if __name__ == "__main__":

    # Let's build our model
    train(1)
    print('Finished Training')

    # Test which classes performed well
    testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    # model = Network()
    path = "HPModel.pth"
    model.load_state_dict(torch.load(path))
    print(testAccuracy())

    # Test with batch of images
    testBatch()