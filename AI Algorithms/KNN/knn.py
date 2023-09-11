import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class KNearestNeighbors:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        X = self.data.iloc[:, [2, 3]].values
        y = self.data.iloc[:, 4].values

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        self.X_train = X_train
        self.y_train = y_train

        self.X_test = X_test
        self.y_test = y_test

    def euclidean_distance(self, instance1, instance2):
        distance = np.sqrt(np.sum((instance1 - instance2) ** 2))
        return distance

    def get_neighbors(self, X, y, k):
        distances = []
        for i in range(X.shape[0]):
            dist = self.euclidean_distance(y, X[i])
            distances.append(dist)
        k_indices = np.argsort(distances)[:int(k)]
        neighbors = self.y_train[k_indices]
        return neighbors

    def get_class(self, neighbors):
        most_common = Counter(neighbors).most_common()
        return most_common[0][0]

    def predict(self, k, X):
        predictions = []
        for i in range(X.shape[0]):
            neighbors = self.get_neighbors(self.X_train, X[i], k)
            predicted_class = self.get_class(neighbors)
            predictions.append(predicted_class)
        return np.array(predictions)

    def evaluate(self, k):
        predictions = self.predict(k, self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy

    def plot_decision_boundary(self, k):
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold = ListedColormap(['#FF0000', '#05bc00'])

        # Plot test set
        plt.figure()
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cmap_bold)
        plt.xlim(self.X_test[:, 0].min() - 0.1, self.X_test[:, 0].max() + 0.1)
        plt.ylim(self.X_test[:, 1].min() - 0.1, self.X_test[:, 1].max() + 0.1)
        plt.title(f'Decision boundry for k: {k}')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')

        # Plot decision boundary
        x_min, x_max = self.X_test[:, 0].min() - 1, self.X_test[:, 0].max() + 1
        y_min, y_max = self.X_test[:, 1].min() - 1, self.X_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        Z = self.predict(k, np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot predicted points
        predictions = self.predict(k, self.X_test)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=predictions, cmap=cmap_bold)
        plt.show()

###################################################################################################
# Train the KNN
knn = KNearestNeighbors("Social_Network_Ads.csv")
k = 5

###################################################################################################
# Evaluate the accuracy
accuracy = knn.evaluate(k)
print(f"Accuracy with k={k}: {accuracy}")

###################################################################################################
# Plot the result
knn.plot_decision_boundary(k)
