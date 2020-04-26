
import pandas as pd

from  sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split,  LeaveOneOut
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from texttable import Texttable

link = "Senzori_dijagnoza_svi.csv"
names = ['Pacijent','Dijagnoza','sensor 1','sensor 2','sensor 3','sensor 4']
dataset = pd.read_csv(link, names=names)
sensors = dataset.iloc[3:, 1:5].values
dijagnoza = dataset.iloc[3:, 0:1].values



normalized_X = preprocessing.normalize(sensors) #normalizovani podaci

encoder = LabelEncoder()
encoded_target = encoder.fit_transform(dijagnoza)  
#print(encoded_target)

#LeaveOneOut
#Omogućuje podelu podatakaza treniranje i testiranje
#Svaki uzorak-sample  koristi se jednom kao test set(singleton), 
#dok preostali uzorci čine set za trening
leaveOneOut = LeaveOneOut() 
leaveOneOut.get_n_splits(sensors)

#nizovi za preciznost algoritma
LR_acc = []
DF_acc = []
RF_acc = []
SVM_acc= []
NB_acc = []
KNN_acc= []

##########TEST=1################OSTATAK TRENING################################
###############################################################################

#######################BEZ NORMALIZACIJE#######################################
for treniranje_indeks, testiranje_indeks in leaveOneOut.split(sensors):
    x_train, x_test = sensors[treniranje_indeks], sensors[testiranje_indeks] #izdvajanje fičera
    y_train, y_test = encoded_target[treniranje_indeks], encoded_target[testiranje_indeks] #uzeće lejbele
    
    #pozivanje modela i racunanje tacnosti
    logReg = LogisticRegression(random_state=0,penalty='l2').fit(x_train, y_train)
    pred_y = logReg.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    LR_acc.append(acc)
#    print(acc)
    
    decisionTree = DecisionTreeClassifier(random_state=0, criterion = 'entropy', splitter = 'best', min_samples_split=3, min_samples_leaf=1).fit(x_train, y_train)
    pred_y = decisionTree.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    DF_acc.append(acc)
    
    
    randomForest = RandomForestClassifier(max_depth=10, random_state=0).fit(x_train, y_train)
    pred_y = randomForest.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    RF_acc.append(acc)
    
    svm = SVC().fit(x_train, y_train)
    pred_y = svm.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    SVM_acc.append(acc)
    
    nb =  MultinomialNB().fit(x_train, y_train)
    pred_y = nb.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    NB_acc.append(acc)
    
    knn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pred_y = knn.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    KNN_acc.append(acc)
    
print("\nBezNormalizacije\n")    
LR_bN_acc = sum(LR_acc)/len(LR_acc)
print("LR: ", LR_bN_acc)

DT_bN_acc = sum(DF_acc)/len(DF_acc)
print("DT: ", DT_bN_acc)

RF_bN_acc = sum(RF_acc)/len(RF_acc)
print("RF: ", RF_bN_acc )

SVM_bN_acc = sum(SVM_acc)/len(SVM_acc)
print("SVM: ", SVM_bN_acc)

NB_bN_acc = sum(NB_acc)/len(NB_acc)
print("NB: ", NB_bN_acc)

KNN_bN_acc = sum(KNN_acc)/len(KNN_acc)
print("KNN: ", KNN_bN_acc)



###############################################################################
###########################MinMax##############################################
###############################################################################

scaler_min_max = MinMaxScaler()
min_max_data = scaler_min_max.fit_transform(sensors) #za fičere

for treniranje_indeks, testiranje_indeks in leaveOneOut.split(min_max_data):
    x_train, x_test = min_max_data[treniranje_indeks], min_max_data[testiranje_indeks]
    y_train, y_test = encoded_target[treniranje_indeks], encoded_target[testiranje_indeks]
    
    logReg = LogisticRegression(random_state=0,penalty='l2').fit(x_train, y_train)
    pred_y = logReg.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    LR_acc.append(acc)
    
    decisionTree = DecisionTreeClassifier(random_state=0, criterion = 'entropy', splitter = 'best', min_samples_split=3, min_samples_leaf=1).fit(x_train, y_train)
    pred_y = decisionTree.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    DF_acc.append(acc)
    
    randomForest = RandomForestClassifier(max_depth=10, random_state=0).fit(x_train, y_train)
    pred_y = randomForest.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    RF_acc.append(acc)
    
    svm = SVC().fit(x_train, y_train)
    pred_y = svm.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    SVM_acc.append(acc)
    
    nb =  MultinomialNB().fit(x_train, y_train)
    pred_y = nb.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    NB_acc.append(acc)
    
    knn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pred_y = knn.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    KNN_acc.append(acc)



print("\nMinMax\n")
LR_MinMax_acc = sum(LR_acc)/len(LR_acc)
print("LR: ", LR_MinMax_acc)

DT_MinMax_acc = sum(DF_acc)/len(DF_acc)
print("DT: ", DT_MinMax_acc)

RF_MinMax_acc = sum(RF_acc)/len(RF_acc)
print("RF: ", RF_MinMax_acc)

SVM_MinMax_acc = sum(SVM_acc)/len(SVM_acc)
print("SVM: ", SVM_MinMax_acc)

NB_MinMax_acc = sum(NB_acc)/len(NB_acc)
print("NB: ", NB_MinMax_acc)

KNN_MinMax_acc = sum(KNN_acc)/len(KNN_acc)
print("KNN: ", KNN_MinMax_acc)



###############################################################################
###################MaxAbs######################################################
###############################################################################
scaler_max_abs = MaxAbsScaler()
max_abs_data = scaler_max_abs.fit_transform(sensors)

for treniranje_indeks, testiranje_indeks in leaveOneOut.split(max_abs_data):
    x_train, x_test = max_abs_data[treniranje_indeks], max_abs_data[testiranje_indeks]
    y_train, y_test = encoded_target[treniranje_indeks], encoded_target[testiranje_indeks]
    
    logReg = LogisticRegression(random_state=0,penalty='l2').fit(x_train, y_train)
    pred_y = logReg.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    LR_acc.append(acc)
    
    decisionTree = DecisionTreeClassifier().fit(x_train, y_train)
    pred_y = decisionTree.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    DF_acc.append(acc)
    
    randomForest = RandomForestClassifier(max_depth=8, random_state=0).fit(x_train, y_train)
    pred_y = randomForest.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    RF_acc.append(acc)
    
    svm = SVC().fit(x_train, y_train)
    pred_y = svm.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    SVM_acc.append(acc)
    
    nb =  MultinomialNB().fit(x_train, y_train)
    pred_y = nb.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    NB_acc.append(acc)
    
    knn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pred_y = knn.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    KNN_acc.append(acc)


print("\nMaxAbs\n")
LR_MaxAbs_acc = sum(LR_acc)/len(LR_acc)
print("LR: ", LR_MaxAbs_acc)

DT_MaxAbs_acc = sum(DF_acc)/len(DF_acc)
print("DT: ", DT_MaxAbs_acc)

RF_MaxAbs_acc = sum(RF_acc)/len(RF_acc)
print("RF: ", RF_MaxAbs_acc)

SVM_MaxAbs_acc = sum(SVM_acc)/len(SVM_acc)
print("SVM: ", SVM_MaxAbs_acc)

NB_MaxAbs_acc = sum(NB_acc)/len(NB_acc)
print("NB: ", NB_MaxAbs_acc)

KNN_MaxAbs_acc = sum(KNN_acc)/len(KNN_acc)
print("KNN: ", KNN_MaxAbs_acc)

###############################################################################
###################NORMALIZACIJA###############################################
###############################################################################
norm_data = preprocessing.normalize(sensors, axis=0, norm= 'l2')

for treniranje_indeks, testiranje_indeks in leaveOneOut.split(norm_data):
    x_train, x_test = norm_data[treniranje_indeks], norm_data[testiranje_indeks]
    y_train, y_test = encoded_target[treniranje_indeks], encoded_target[testiranje_indeks]
    
    logReg = LogisticRegression(random_state=0,penalty='l2').fit(x_train, y_train)
    pred_y = logReg.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    LR_acc.append(acc)
    
    decisionTree = DecisionTreeClassifier().fit(x_train, y_train)
    pred_y = decisionTree.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    DF_acc.append(acc)
    
    randomForest = RandomForestClassifier(max_depth=8, random_state=0).fit(x_train, y_train)
    pred_y = randomForest.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    RF_acc.append(acc)
    
    svm = SVC().fit(x_train, y_train)
    pred_y = svm.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    SVM_acc.append(acc)
    
    nb =  MultinomialNB().fit(x_train, y_train)
    pred_y = nb.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    NB_acc.append(acc)
    
    knn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pred_y = knn.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    KNN_acc.append(acc)



print("\nNormalizacija\n")
LR_Norm_acc = sum(LR_acc)/len(LR_acc)
print("LR: ", LR_Norm_acc)

DT_Norm_acc = sum(DF_acc)/len(DF_acc)
print("DT: ", DT_Norm_acc)

RF_Norm_acc = sum(RF_acc)/len(RF_acc)
print("RF: ", RF_Norm_acc)

SVM_Norm_acc = sum(SVM_acc)/len(SVM_acc)
print("SVM: ", SVM_Norm_acc)

NB_Norm_acc = sum(NB_acc)/len(NB_acc)
print("NB: ", NB_Norm_acc)

KNN_Norm_acc = sum(KNN_acc)/len(KNN_acc)
print("KNN: ", KNN_Norm_acc)

###############################################################################
####################STANDARDIZACIJA############################################
###############################################################################
scaler_standard = StandardScaler()
standard_data = scaler_standard.fit_transform(sensors)

for treniranje_indeks, testiranje_indeks in leaveOneOut.split(standard_data):
    x_train, x_test = standard_data[treniranje_indeks], standard_data[testiranje_indeks]
    y_train, y_test = encoded_target[treniranje_indeks], encoded_target[testiranje_indeks]
    
    logReg = LogisticRegression(random_state=0,penalty='l2').fit(x_train, y_train)
    pred_y = logReg.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    LR_acc.append(acc)
    
    decisionTree = DecisionTreeClassifier().fit(x_train, y_train)
    pred_y = decisionTree.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    DF_acc.append(acc)
    
    randomForest = RandomForestClassifier(max_depth=8, random_state=0).fit(x_train, y_train)
    pred_y = randomForest.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    RF_acc.append(acc)
    
    svm = SVC().fit(x_train, y_train)
    pred_y = svm.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    SVM_acc.append(acc)
    
#javlja gresku, ne moze da radi  sa negativnim brojevima    
#    nb =  MultinomialNB().fit(x_train, y_train)
#    pred_y = nb.predict(x_test)
#    acc = accuracy_score(y_test, pred_y)
#    NB_acc.append(acc)
    
    knn = KNeighborsClassifier(n_neighbors=3).fit(x_train, y_train)
    pred_y = knn.predict(x_test)
    acc = accuracy_score(y_test, pred_y)
    KNN_acc.append(acc)



print("\nStandardizacija\n")
LR_S_acc = sum(LR_acc)/len(LR_acc)
print("LR: ", LR_S_acc)

DT_S_acc = sum(DF_acc)/len(DF_acc)
print("DT: ", DT_S_acc)

RF_S_acc = sum(RF_acc)/len(RF_acc)
print("RF: ", RF_S_acc)

SVM_S_acc = sum(SVM_acc)/len(SVM_acc)
print("SVM: ", SVM_S_acc)

#NB_S_acc = sum(NB_acc)/len(NB_acc)
print("NB: ", '/')

KNN_S_acc = sum(KNN_acc)/len(KNN_acc)
print("KNN: ", KNN_S_acc)

#tabelica
t = Texttable()
t.add_rows([
            ['Algoritmi\nTehnike ', "LR", "DT", "RF", "NB", "SVM", "KNN"],
            ['BezNormalizacije', LR_bN_acc,      DT_bN_acc,     RF_bN_acc,     NB_bN_acc,     SVM_bN_acc,     KNN_bN_acc],
            ['MinMax' , LR_MinMax_acc,  DT_MinMax_acc, RF_MinMax_acc, NB_MinMax_acc, SVM_MinMax_acc, KNN_MinMax_acc],
            ['MaxAbs' , LR_MaxAbs_acc,  DT_MaxAbs_acc, RF_MaxAbs_acc, NB_MaxAbs_acc, SVM_MaxAbs_acc, KNN_MaxAbs_acc],
            ['Normalizacija' , LR_Norm_acc,  DT_Norm_acc, RF_Norm_acc, NB_Norm_acc, SVM_Norm_acc, KNN_Norm_acc],
            ['Standardizacija' , LR_S_acc,  DT_S_acc, RF_S_acc, '  /', SVM_S_acc, KNN_S_acc]
            ])
    
print(t.draw())


