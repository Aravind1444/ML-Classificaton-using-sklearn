import pandas as pd 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import svm
import matplotlib.pyplot as plt
import os

def experiment():
    global X
    global y

    print('')

    print("The available scaling methods are : ")
    print('1. StandarScaler ')
    print('2. Normalizer')
    print('3. MinMaxScaler')
    print('')
    scaler = input("Which scaler do you want to use : ") #Taking input from the user to select the scaler
    print('')
    if scaler == '1':                       # StandardScaler
        scaler = StandardScaler()
        print('Scaler Selected : ', scaler)
        scaler = scaler.fit(X)
        scaled_X = scaler.transform(X)
    elif scaler == '2':                         # Normalizer
        transformer = Normalizer().fit(X)
        scaled_X = transformer.transform(X)
    elif scaler == '3':                             # MinMaxScaler
        scaler = MinMaxScaler()
        scaler = scaler.fit(X)
        scaled_X = scaler.transform(X)



    print('')
    print('Scaled Data : ')                              # Printing the properties of the scaled data : scaled_X
    print('')
    print(scaled_X[0:1])
    print('')
    print('--- Scaled Data Properties ---')
    print('')
    print('Mean : ',scaled_X.mean(),' Var : ', scaled_X.var()) 
    print('')
    print('#######################################')

    # scaled_data1 = pd.DataFrame(scaled_data)
    print('')
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.3, random_state=42)
    
    print('The available classifiers are : ')
    print('1. SVM classifier with automatic Kernel Selection')
    print('2. SVM classifier with Polynomial Kernel')
    print('3. Random Forest classifier ')
    print('')
    classifier = input("Which classifier do you want to use : ")        # getting input from the user to choose the classifier
    print('')
    if classifier == '1':                                         # SVM with kernel=auto is choosed
        clf = svm.SVC(gamma='auto')
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("")
        print('Score : ',score)

    elif classifier == '2':                                       # SVM with Kernel = poly with degree 2 is selected
        clf = svm.SVC(gamma = 1, kernel = 'poly', C = 0.05, degree = 2)
        clf = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("")
        print('Score is : ', score)

    elif classifier == '3' :                                 # Random Forest classifier is selected for classification
        print("")
        scaled_X, y = make_classification(n_features=30)
        clf = RandomForestClassifier()
        clf = clf.fit(scaled_X, y)
        print("")
        print('Classifier is selected and training being done.')
        #print(clf.feature_importances_)
        print("")
        score = clf.score(X_test, y_test)
        print("The score of the model is : ", score)

    print("")
    print("")

    print("#######################################")                    # End!
    print('')


os.system('clear')
print('Initializing')
print('')
data = pd.read_csv("/home/aravind/Desktop/Project01/data/data.csv")
print('#######################################')
print('')
print('Data Loaded')
print(data.head())

X = data.iloc[:,2:33] #using iloc to specify the range which should be assigned to the variable.
y = data.iloc[:,1] # ':' specifies that, all rows should be selected and the other specifies the number of rows and columns.


while(True):
    experiment()
