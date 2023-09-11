import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        X = self.data.iloc[:, [2, 3]].values
        y = self.data.iloc[:, 4].values

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        
        self.X_train = X_train
        self.y_train = y_train.reshape(-1, 1)

        self.X_test = X_test
        self.y_test = y_test.reshape(-1, 1)

    def calculate_entropy(self, y):
        unique_labels = np.unique(y)
        entropy = 0

        for label in unique_labels:
            label_indices = (y == label)
            label_probability = np.sum(label_indices) / y.shape[0]
            entropy -= label_probability * np.log2(label_probability)
        return entropy
    
    def calculate_information_gain(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        
        left_entropy = self.calculate_entropy(y[left_indices])
        right_entropy = self.calculate_entropy(y[right_indices])
        
        parent_entropy = self.calculate_entropy(y)
        left_weight = sum(left_indices) / y.shape[0]
        right_weight = sum(right_indices) / y.shape[0]
        
        information_gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
        return information_gain

    
    def find_best_split(self, X, y):
        best_info_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            unique_values = np.unique(np.sort(feature_values))

            for i in range(1, len(unique_values)):
                threshold = (unique_values[i-1] + unique_values[i]) / 2
                info_gain = self.calculate_information_gain(X, y, feature_index, threshold)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature= feature_index
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def create_leaf_node(self, y):
        unique_labels, counts = np.unique(y, return_counts=True)
        majority_class = unique_labels[np.argmax(counts)]
        return {'class': majority_class}
    
    def create_decision_tree(self, X, y):
        if np.unique(y).shape[0] == 1:
            return self.create_leaf_node(y)
        
        if X.shape[0] == 0:
            return self.create_leaf_node(y)
        
        best_feature, best_threshold = self.find_best_split(X, y)
        if best_feature is None:
            return self.create_leaf_node(y)
        
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        tree = {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': self.create_decision_tree(X[left_indices], y[left_indices]),
            'right': self.create_decision_tree(X[right_indices], y[right_indices])
        }
        
        return tree
    
    def fit(self):
        self.tree = self.create_decision_tree(self.X_train, self.y_train)
    
    def predict_sample(self, sample, node):
        if 'class' in node:
            return node['class']
        
        feature_index = node['feature_index']
        threshold = node['threshold']
        
        if sample[feature_index] <= threshold:
            return self.predict_sample(sample, node['left'])
        else:
            return self.predict_sample(sample, node['right'])
    
    def predict(self, X):
        predictions = []
        for sample in X:
            prediction = self.predict_sample(sample, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

    def plot_decision_tree(self):
        cmap = ListedColormap(['red', 'green'])
        # Plot the training set results
        X_set, y_set = tree.X_train, tree.y_train.flatten()
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                            np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
        plt.contourf(X1, X2, tree.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=cmap)
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=cmap(i), label=j)
        plt.title('Decision Tree (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()
        
###################################################################################################
# Train the decision tree
tree = DecisionTree("Social_Network_Ads.csv")
tree.fit()
# Make predictions on test set
y_pred = tree.predict(tree.X_test)
print(y_pred)

###################################################################################################
# Evaluate the accuracy
accuracy = accuracy_score(tree.y_test, y_pred)
print("Accuracy:", accuracy)

###################################################################################################
# Plot the tree
tree.plot_decision_tree()