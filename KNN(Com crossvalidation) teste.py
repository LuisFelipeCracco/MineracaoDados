# Initial imports
import itertools
import pandas as pd
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
import seaborn as sns
from typing import Tuple

# Calculate distance between two points
def minkowski_distance(a, b, p=1):    
    # Store the number of dimensions
    dim = len(a)    
    # Set initial distance to 0
    distance = 0
    
    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d])**p
        
    distance = distance**(1/p)    
    return distance


def knn_predict(X_train, X_test, y_train, y_test, k, p):    
    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)
        
        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=y_train.index)
        
        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]
        
        # Append prediction to output list
        y_hat_test.append(prediction)
        
    return y_hat_test


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure()
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')    

def plot_confusion_matrix(actual_classes : np.array, predicted_classes : np.array, sorted_labels : list):

    matrix = confusion_matrix(actual_classes, predicted_classes, labels=sorted_labels)
    
    plt.figure(figsize=(12.8,6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')

    plt.show()

def cross_val_predict(model, kfold : KFold, X : np.array, y : np.array) -> Tuple[np.array, np.array]:

    model_ = cp.deepcopy(model)
    
    no_classes = len(np.unique(y))
    
    actual_classes = np.empty([0], dtype=int)
    predicted_classes = np.empty([0], dtype=int)
    predicted_proba = np.empty([0, no_classes]) 

    for train_ndx, test_ndx in kfold.split(X):

        train_X, train_y, test_X, test_y = X[train_ndx], y[train_ndx], X[test_ndx], y[test_ndx]

        actual_classes = np.append(actual_classes, test_y)

        model_.fit(train_X, train_y)
        predicted_classes = np.append(predicted_classes, model_.predict(test_X))

        try:
            predicted_proba = np.append(predicted_proba, model_.predict_proba(test_X), axis=0)
        except:
            predicted_proba = np.append(predicted_proba, np.zeros((len(test_X), no_classes), dtype=float), axis=0)

    return actual_classes, predicted_classes, predicted_proba


def main():
    input_file = 'hcvdatclearnumerico01.csv'
    names = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
    features = ['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
    target = 'Category'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    
    # Separate X and y data
    X = df.drop(target, axis=1)
    print(X)
    y = df[target].round(decimals=0)
    print(y)  
    print("Total samples: {}".format(X.shape[0]))
    
    # Split the data - 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    print("Total train samples: {}".format(X_train.shape[0]))
    print("Total test  samples: {}".format(X_test.shape[0]))

    # Scale the X data using Z-score
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    # STEP 1 - TESTS USING knn classifier write from scratch    
    # Make predictions on test dataset using knn classifier
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=7, p=2)

    # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    print("Acurracy K-NN from scratch: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from scratch: {:.2f}%".format(f1))

    # # Get test confusion matrix
    # cm = confusion_matrix(y_test, y_hat_test)        
    # plot_confusion_matrix(cm, ['Blood Donor','Hepatits'], False, "Confusion Matrix - K-NN")      
    # plot_confusion_matrix(cm, ['Blood Donor','Hepatits'], True, "Confusion Matrix - K-NN normalized")  

    # STEP 2 - TESTS USING knn classifier from sk-learn
    knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
    print("Scores do Cross Validation: ")
    print(scores)
    print("M??dia dos Scores do Cross Validation: ")
    print(scores.mean())
    # choose k between 1 to 31
    k_range = range(1, 31)
    k_scores = []
    actual_classes, predicted_classes, _ = cross_val_predict(knn, kfold, X.to_numpy(), y.to_numpy())
    plot_confusion_matrix(actual_classes, predicted_classes, ["Blood Donor", "Hepatits"])
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=kfold, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    plt.figure()
    plt.plot(k_range, k_scores)
    plt.title("KNN com Cross Validation Variando K")
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')

    knn.fit(X_train, y_train)
    y_hat_test = knn.predict(X_test)

     # Get test accuracy score
    accuracy = accuracy_score(y_test, y_hat_test)*100
    f1 = f1_score(y_test, y_hat_test,average='macro')
    print("Acurracy K-NN from sk-learn: {:.2f}%".format(accuracy))
    print("F1 Score K-NN from sk-learn: {:.2f}%".format(f1))


    # Get test confusion matrix    
    # cm = confusion_matrix(y_test, y_hat_test)        
    # plot_confusion_matrix(cm, ['Blood Donor','Hepatits'], False, "Confusion Matrix - K-NN sklearn")      
    # plot_confusion_matrix(cm, ['Blood Donor','Hepatits'], True, "Confusion Matrix - K-NN sklearn normalized" )  
    # plt.show()


if __name__ == "__main__":
    main()