import matplotlib.pyplot as plt
from pandas.core.common import flatten
import numpy as np
import random

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import utils
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.autograd import Variable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import glob

from torch.utils.data import DataLoader


# Custom pre-processing image transformations learned in class, comment out in MyClass to use
def highPass(X):
    X = cv2.resize(X, (int(X.shape[1] * 0.8), int(X.shape[0] * 0.8)))
    return X - cv2.GaussianBlur(X, (21, 21), 3) + 127


# function does not work
def threshold(X):
    X = cv2.resize(X, (int(X.shape[1] * 0.8), int(X.shape[0] * 0.8)))
    return cv2.threshold(X, 100, 255, cv2.THRESH_BINARY)


# transformations
train_transforms = A.Compose([
    A.SmallestMaxSize(max_size=350),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

test_transforms = A.Compose([
    A.SmallestMaxSize(max_size=350),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
])

train_data_path = 'dataset2-master\dataset2-master\images\TRAIN'
test_data_path = 'dataset2-master\dataset2-master\images\TEST'

train_image_paths = []  #to store image paths in list
classes = []  #to store class values

# creating paths to images
for data_path in glob.glob(train_data_path + '\*'):
    train_image_paths.append(glob.glob(data_path + '\*'))

train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])

# creating paths
test_image_paths = []
for data_path in glob.glob(test_data_path + '\*'):
    test_image_paths.append(glob.glob(data_path + '\*'))

test_image_paths = list(flatten(test_image_paths))

batch_size = 5

classes = ('EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL')

idx_to_class = {i: j for i, j in enumerate(classes)}
class_to_idx = {value: key for key, value in idx_to_class.items()}


# creating our dataset class
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
        image = highPass(image)
        # image = threshold(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


train_dataset = MySet(train_image_paths, train_transforms)
test_dataset = MySet(test_image_paths, test_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Network class, with 3 convolution layers, 3 relu layers, 3, batchnorm layers, a maxpool layer, and a linear output layer
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.c1 = nn.Conv2d(in_channels=3,
                            out_channels=12,
                            kernel_size=5,
                            stride=1,
                            padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.p = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(in_channels=12,
                            out_channels=24,
                            kernel_size=5,
                            stride=1,
                            padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.c3 = nn.Conv2d(in_channels=24,
                            out_channels=24,
                            kernel_size=5,
                            stride=1,
                            padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.fc = nn.Linear(930240, 4)

    def forward(self, input):
        output = F.relu(self.bn1(self.c1(input)))
        output = self.p(output)
        output = F.relu(self.bn2(self.c2(output)))
        output = F.relu(self.bn3(self.c3(output)))
        output = output.view(-1, 930240)
        output = self.fc(output)

        return output


# Create model
model = Network()

# Create optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


def saveModel():
    path = "./hpRaw.pth"
    torch.save(model.state_dict(), path)


# Returns accuracy on train set
def testAccuracy():

    model.eval()
    accuracy = 0.0
    total = 0.0

    count = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            count += 1
            if count == 30:
                break

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return (accuracy)


# training function
def train(num_epochs):

    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # extract the loss value
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        accuracy = testAccuracy()
        print(
            'For epoch', epoch + 1,
            'the test accuracy over the whole test set is %d %%' % (accuracy))

        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():

    images, labels = next(iter(test_loader))

    imageshow(utils.make_grid(images))

    print('Real labels: ',
          ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ',
          ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))


if __name__ == "__main__":

    # Train model
    train(1)
    print('Finished Training')

    # Test which classes performed well
    testAccuracy()

    # Load model
    model = Network()
    path = "hpRaw.pth"
    model.load_state_dict(torch.load(path))
    print(testAccuracy())

    testBatch()