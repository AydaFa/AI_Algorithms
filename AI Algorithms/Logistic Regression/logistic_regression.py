import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

class LogisticRegression:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

        X = self.data.iloc[:, [2, 3]].values
        y = self.data.iloc[:, 4].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        # Normalize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train = X_train.T
        self.y_train = y_train.reshape(1, X_train.shape[0])

        self.X_test = X_test.T
        self.y_test = y_test.reshape(1, X_test.shape[0])

    def model(self, epochs=1000, learning_rate=0.01):
        m = self.X_train.shape[1]
        n = self.X_train.shape[0]
        W = np.zeros((n, 1))
        B = 0

        cost_list = []

        for epoch in range(epochs):
            Z = np.dot(W.T, self.X_train) + B
            y_pred = self.sigmoid(Z)

            cost_function = - (1 / m) * np.sum(self.y_train * np.log(y_pred) + (1 - self.y_train) * np.log(1 - y_pred))

            # Gradients
            dW = (1 / m) * np.dot(self.X_train, (y_pred - self.y_train).T)
            dB = (1 / m) * np.sum(y_pred - self.y_train)
            W -= learning_rate * dW
            B -= learning_rate * dB

            cost_list.append(cost_function)

            if epoch % (epochs / 10) == 0:
                print("Cost after", epoch, "epochs:", cost_function)

        return W, B, cost_list

    def predict(self, X):
        Z = np.dot(self.W.T, X) + self.B
        y_pred = self.sigmoid(Z)
        return np.round(y_pred)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def plot_cost_iterations(self, cost_list):
        plt.plot(cost_list)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Iterations')
        plt.show()

    def evaluate_model(self, y_pred):
        cm = confusion_matrix(self.y_test[0], y_pred[0])
        sns.heatmap(cm, annot=True)
        plt.xlabel('Predicted classes')
        plt.ylabel('True classes')
        plt.title('Confusion matrix')
        plt.show()

        accuracy = accuracy_score(self.y_test[0], y_pred[0])
        print('Accuracy:', accuracy)

###################################################################################################
# Train the Logistic Regression
logistic = LogisticRegression("Social_Network_Ads.csv")
W, B, cost_list = logistic.model()

###################################################################################################
# Plot cost vs iterations
logistic.plot_cost_iterations(cost_list)

###################################################################################################
# Make predictions on test set
logistic.W = W
logistic.B = B
y_pred = logistic.predict(logistic.X_test)
logistic.evaluate_model(y_pred)
