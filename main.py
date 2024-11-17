import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from torchvision import transforms

#load data
labeled_images = np.load("data/labeled_images.npy")
labeled_digits = np.load("data/labeled_digits.npy")
autograder_images = np.load("data/autograder_images.npy")

#data parameters
print(labeled_images.shape)
print(labeled_digits.shape)

# digits, counts = np.unique(labeled_digits, return_counts=True)
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(digits, counts, width=1, edgecolor="white", linewidth=0.7)
# ax.set_title('Distribution of Digits', fontsize=16)
# ax.set_xlabel('Digits', fontsize=14)
# ax.set_ylabel('Frequency', fontsize=14)
#
# labeled_digits[0:100]
#
# #data visualization
# num_images = 10
# plt.figure(figsize=(10, 2))
#
# for i in range(num_images):
#     image = labeled_images[i]
#     plt.subplot(1, num_images, i + 1)
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
# plt.show()

#data preprocessing
X = labeled_images.reshape(labeled_images.shape[0], -1)
print(X.shape)
#print(X.mean(axis=0))
#print(X.std(axis=0))
scaler = preprocessing.StandardScaler().fit(X)
X_processed = scaler.transform(X)
print(X_processed.shape)
means = X_processed.mean(axis=0)
stds = X_processed.std(axis=0)
print(means)
print(stds)
print(means.mean())
print(stds.mean())

#train and validate
#X = labeled_images.reshape(labeled_images.shape[0], -1)
Y = labeled_digits
X_train, X_test, Y_train, Y_test = train_test_split(X_processed, Y, test_size=0.2, random_state=42)

svm_clf = SVC(kernel='rbf')

cv_scores = cross_val_score(svm_clf, X_train, Y_train, cv=10, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Average Accuracy:", cv_scores.mean())

#test
svm_clf.fit(X_train, Y_train)
test_accuracy = svm_clf.score(X_test, Y_test)
print("Test set accuracy:", test_accuracy)