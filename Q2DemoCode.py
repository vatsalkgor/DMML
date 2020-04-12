# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:02:54 2020

@author: rpear
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
np.random.seed(1)

""" Example based on sklearn's docs """
mnist = fetch_openml('mnist_784')
# rescale the data, use the traditional train/test split
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:1000], X[1000:]
y_train, y_test = y[:1000], y[1000:]

mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=5, alpha=1e-4,
                    solver='sgd', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.01)


N_TRAIN_SAMPLES = X_train.shape[0]
N_EPOCHS = 25
N_BATCH = 64
N_CLASSES = np.unique(y_train)

scores_train = []
scores_test = []
mlploss = []


# EPOCH
epoch = 0
while epoch < N_EPOCHS:
    print('epoch: ', epoch)
    # SHUFFLING
    random_perm = np.random.permutation(X_train.shape[0])
    mini_batch_index = 0
    while True:
        # MINI-BATCH
        indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
        mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
        mini_batch_index += N_BATCH

        if mini_batch_index >= N_TRAIN_SAMPLES:
            break

    # SCORE TRAIN
    scores_train.append(1-mlp.score(X_train, y_train))
    
     # SCORE TEST
    scores_test.append(1-mlp.score(X_test, y_test))
    
    # compute loss
    
    mlploss.append(mlp.loss_)
    epoch += 1

""" Plot """
fig, ax = plt.subplots(3, sharex=True)
ax[0].plot(scores_train)
ax[0].set_title('Train Error')
ax[1].plot(mlploss)
ax[1].set_title('Train Loss')
ax[2].plot(scores_test)
ax[2].set_title('Test Error')
fig.suptitle("Error vs Loss over epochs", fontsize=14)
fig.savefig('C:/Users/rpear/OneDrive/Apps/Documents/LossCurve.png')
plt.show()

