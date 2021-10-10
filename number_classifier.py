import csv
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
from tqdm import trange


VERSION = 2
MODEL_DIR = "digit_classifier_nets"

IMAGE_SIZE = (20, 15)
OUTPUT_SIZE = 11
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.7
NUM_EPOCHS = 100

TRAIN_RATIO = 1
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64

TRANSFORMATIONS = None
TRAIN_ROOT_DIR = "training_data"
TRAIN_DATA_DIR = "digits"
TRAIN_DATA_CSV = "data_map.csv"
TRAIN_DIRECTORY = os.path.join(TRAIN_ROOT_DIR, TRAIN_DATA_DIR)

CLASS_NAMES = list(map(str, list(range(10)))) + list("?")


def read_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def create_dict_from_list(tuple_list):
    dict_list = {}
    for y, x in tuple_list:
        dict_list.setdefault(x, []).append(y)
    return dict_list


def show_samples(data_loader):
    for batch in data_loader:
        for image, label in zip(batch["image"], batch["label"]):
            plt.imshow(image[0], cmap='gray')
            plt.title(label)
            plt.show()
        break


# Plot sample data for each class
def plot_samples(label_dict, num_samples=10):
    num_classes = len(label_dict.keys())
    fig, ax = plt.subplots(num_classes, num_samples, figsize=(20, 60))

    for row, class_name in enumerate(sorted(label_dict.keys())):

        image_file_list = label_dict[class_name]

        random.shuffle(image_file_list)
        samples = image_file_list[0:num_samples]

        for col in range(num_samples):
            if col == 0:
                ax[row][col].set_ylabel(f"Class: {class_name}")
            image_path = os.path.join(TRAIN_DIRECTORY, f"{samples[col]}.png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
            ax[row][col].imshow(image)

    plt.tight_layout()
    plt.show()


# Data statistics for train and test sets
def plot_data_statistics():
    train_label_file = os.path.join(TRAIN_ROOT_DIR, TRAIN_DATA_CSV)
    train_list = read_csv(train_label_file)
    train_dict = create_dict_from_list(train_list)

    total_train = 0
    for k in sorted(train_dict.keys()):
        total_train += len(train_dict[k])

    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.title(f"Training set ({total_train} images)")
    plt.bar(sorted(train_dict.keys()), [len(train_dict[k]) for k in sorted(train_dict.keys())])
    plt.show()


# Plot samples from the training set
def plot_training_samples():
    train_label_file = os.path.join(TRAIN_ROOT_DIR, TRAIN_DATA_CSV)
    train_list = read_csv(train_label_file)
    train_dict = create_dict_from_list(train_list)
    plot_samples(train_dict, num_samples=10)


class DigitDataset(Dataset):
    def __init__(self,  root_dir, csv_file, data_dir, class_index_map=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, data_dir)
        self.class_index_map = class_index_map
        self.transform = transform

        (unique, counts) = np.unique(np.array(self.data)[:, 1], return_counts=True)
        allowed = int(min(counts) * 1.1)
        c = dict()
        for cn in CLASS_NAMES:
            c[cn] = allowed
        balanced_data = list()
        for dp in self.data:
            if c[dp[1]] > 0:
                c[dp[1]] -= 1
                balanced_data.append(dp)
        self.data = balanced_data

    def __len__(self):
        """
        Calculates the length of the dataset-
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample (dict consisting of an image and its label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the image and labels
        image_path = os.path.join(self.data_dir, f"{self.data[idx][0]}.png")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Shape of the image should be H,W,C where C=1
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        # The label is the index of the class name in the list ['0','1',...,'9','a','b',...'z']
        # because we should have integer labels in the range 0-35 (for 36 classes)
        label = CLASS_NAMES.index(self.data[idx][1])

        sample = {'image': image, 'label': label}

        return sample


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE[0] * IMAGE_SIZE[1], HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc5 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc6 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc7 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc8 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc9 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = torch.tanh(self.fc9(x))

        return x


def calc_accuracy_minibatch(net, data_loader):
    net.eval()
    correct = 0
    with torch.no_grad():
        for sign_dict in data_loader:
            data = sign_dict["image"].float()
            target = sign_dict["label"]
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / len(data_loader.dataset)
    return accuracy


def calc_accuracy_minibatch_detailed(net, data_loader):
    train_label_file = os.path.join(TRAIN_ROOT_DIR, TRAIN_DATA_CSV)
    train_list = read_csv(train_label_file)
    train_dict = create_dict_from_list(train_list)
    total_train = 0
    correct_label = list()
    num_label = list()
    for k in sorted(train_dict.keys()):
        total_train += len(train_dict[k])
        correct_label.append(0)
        num_label.append(0)

    correct_total = 0
    with torch.no_grad():
        for sign_dict in data_loader:
            data = sign_dict["image"].float()
            target = sign_dict["label"]
            output = net(data)
            pred = output.argmax(dim=1, keepdim=True)

            check = pred.eq(target.view_as(pred))
            correct_total += check.sum().item()
            for i, correct in enumerate(target):
                correct_label[correct] += check[i].item()
                num_label[correct] += 1

    print("----------------------")
    print("| Label   Num   Acc  |")
    print("----------------------")
    print(f"| Total {len(data_loader.dataset)}  {correct_total / len(data_loader.dataset):.2f} |")
    for i, k in enumerate(sorted(train_dict.keys())):
        print(f"|   {k}    {num_label[i]}  {correct_label[i] / num_label[i]:.2f}  |")
    print("----------------------")


def train(model, num_epochs: int = NUM_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    model.train()
    losses = []

    for epoch in trange(num_epochs):
        loss_of_epoch = 0.0
        for batch_idx, digit_dict in enumerate(train_dataloader):
            data = digit_dict["image"].float()
            target = digit_dict["label"]

            optimizer.zero_grad()
            output = model.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            loss_of_epoch += loss.item() * data.shape[0]

        losses.append(loss_of_epoch / len(train_dataset))

    return losses


def classify_digit(digit: np.array):
    input = torch.from_numpy(np.expand_dims(digit.astype(np.float), 0)).float()
    target = net.forward(input)
    result = CLASS_NAMES[torch.argmax(target)]
    if result == '?':
        return None
    else:
        return int(result)


net = DigitClassifier()
net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v" + str(VERSION) + ".pt")))


if __name__ == "__main__":
    digit_dataset = DigitDataset(root_dir=TRAIN_ROOT_DIR, csv_file=TRAIN_DATA_CSV, data_dir=TRAIN_DATA_DIR,
                                 transform=TRANSFORMATIONS)
    data_len = len(digit_dataset)
    train_size = int(TRAIN_RATIO * data_len)
    test_size = data_len - train_size
    train_dataset, test_dataset = random_split(digit_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    #test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0)

    net = DigitClassifier()
    # net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "model_v" + str(VERSION) + ".pt")))

    #test_accuracy = calc_accuracy_minibatch(net=net, data_loader=test_dataloader)
    #train_accuracy = calc_accuracy_minibatch(net=net, data_loader=train_dataloader)
    losses = train(net)
    print(losses)
    #print(test_accuracy, train_accuracy)
    calc_accuracy_minibatch_detailed(net=net, data_loader=train_dataloader)
    torch.save(net.state_dict(), os.path.join(MODEL_DIR, ('model_v' + str(VERSION) + '.pt')))
