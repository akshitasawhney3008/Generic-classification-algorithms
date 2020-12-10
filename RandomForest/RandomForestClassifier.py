import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, matthews_corrcoef, f1_score
import pickle

iterations = 4
threshold = 0.5

def getPredictionsGivenThreshold(myMatrix, th):
    myList = []
    for i in range(myMatrix.shape[0]):
        p1 = myMatrix[i, 1]
        if p1>= th:
            myList.append(1)
        else:
            myList.append(0)
    return np.asarray(myList)

def getResults(listOfPredProb, labels):
    zeroProb = []
    oneProb = []
    zeroNpArr = []
    oneNpArr = []
    for predProb in listOfPredProb:
        zeroProb.append(predProb[:, 0])
        oneProb.append(predProb[:, 1])
    for i in range(listOfPredProb[0].shape[0]):
        totZeroProb = zeroProb[0][i] + zeroProb[1][i] + zeroProb[2][i] + zeroProb[3][i]
        finZeroProb = (totZeroProb*1.0)/4
        totOneProb = oneProb[0][i] + oneProb[1][i] + oneProb[2][i] + oneProb[3][i]
        finOneProb = (totOneProb * 1.0) / 4
        zeroNpArr.append(finZeroProb)
        oneNpArr.append(finOneProb)
    matrix = np.column_stack((zeroNpArr, oneNpArr))
    matrixPredictions = getPredictionsGivenThreshold(matrix, threshold)
    precision = precision_score(labels, matrixPredictions)
    recall = recall_score(labels, matrixPredictions)
    auroc = roc_auc_score(labels, matrix[:, 1])
    accuracy = accuracy_score(labels, matrixPredictions)
    matthewsCoeff = matthews_corrcoef(labels, matrixPredictions)
    f1score = f1_score(labels, matrixPredictions)

    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1score))
    print('Accuracy: ' + str(accuracy))
    print('AUROC: ' + str(auroc))
    print('MCC: ' + str(matthewsCoeff))


X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

listOfPredictionProbabilities = []
for i in range(iterations):
    with open('Model_' + str(i) + '.pkl', 'rb') as f:
        model = pickle.load(f)
    predictionProbabilities = model.predict_proba(X_test)
    listOfPredictionProbabilities.append(predictionProbabilities)

getResults(listOfPredictionProbabilities, Y_test)
print('Done')