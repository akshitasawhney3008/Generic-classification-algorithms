import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

column_file = open("pos_train_normal1.txt", 'r')
column_file_read = column_file.readline()
column_names = column_file_read.rstrip('\n').split(',')
column_names_list = []
for c in column_names:
    column_names_list.append(c)

column_names_list.append('Label')
column_names = np.asarray(column_names_list)

pos_train_numpy_array = np.genfromtxt('pos_train_normal1.txt', delimiter=',',skip_header=1)
neg_train_numpy_array = np.genfromtxt('neg_train_normal1.txt', delimiter=',',skip_header=1)
pos_test_numpy_array = np.genfromtxt('pos_test_normal1.txt', delimiter=',',skip_header=1)
neg_test_numpy_array = np.genfromtxt('neg_test_normal1.txt', delimiter=',',skip_header=1)


postrain = np.zeros((pos_train_numpy_array.shape[0],1), dtype=float)
negtrain = np.ones((neg_train_numpy_array.shape[0],1), dtype=float)
postest = np.zeros((pos_test_numpy_array.shape[0],1), dtype=float)
negtest = np.ones((neg_test_numpy_array.shape[0],1), dtype=float)
targetcol_train= np.append(postrain, negtrain ,axis=0)
targetcol_test = np.append(postest, negtest ,axis=0)
targetcol = np.append(postrain, negtrain,axis=0)
targetcol = np.append(targetcol, postest, axis=0)
targetcol = np.append(targetcol, negtest, axis=0)


wholedataset = np.append(pos_train_numpy_array,neg_train_numpy_array, axis=0)
wholedataset = normalize(wholedataset)
wholedataset_test = np.append(pos_test_numpy_array,neg_test_numpy_array,axis=0)
wholedataset_test = normalize(wholedataset_test)
final_data = np.append(wholedataset,wholedataset_test,axis=0)
# wholedataset = np.append(wholedataset,pos_test_numpy_array,axis=0)
# wholedataset = np.append(wholedataset,neg_test_numpy_array,axis=0)

# whole_dataset_train = np.append(pos_train_numpy_array,pos_test_numpy_array, axis=0)
whole_dataset_train = np.append(wholedataset,targetcol_train, axis=1)
# whole_dataset_test = np.append(neg_train_numpy_array,neg_test_numpy_array, axis=0)
whole_dataset_test = np.append(wholedataset_test,targetcol_test, axis=1)

labels = []
targetcol = targetcol.tolist()
for lab in targetcol:
    if lab == [0.0]:
        labels.append('P')
    else:
        labels.append('N')

labels = np.asarray(labels).reshape(-1, 1)
final_data = np.append(final_data,labels,axis=1)
column_names = column_names.reshape(1, -1)
final_data = np.append(column_names,final_data,axis=0)
fmt2 = ",".join(["%s"] + ["%s"] * (final_data.shape[1]-1))


np.savetxt("final_data.csv",final_data,delimiter=",", fmt=fmt2)

X = final_data[1:,:-1]
y = final_data[1:,-1]

lsvc = LinearSVC(C=1.0, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
indexes = model.get_support(indices=True)
print(X_new.shape)

ind = []
for inde in indexes:
    ind.append(inde)
print(ind)

print(column_names[:,ind])