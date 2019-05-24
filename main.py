import pandas as pd 
import numpy as np 
import stats as sts  

#建立一個15*5的二維陣列
post=np.zeros((15,5))
#post
for i in range(15):
    path='data/post/' + str(i+1) + '.csv'
    #將V前1萬5千筆讀出來並分析,skiprows為跳過第一個row
    V = pd.read_csv(path, skiprows=1)['V'][0:20004]
    post[i,0]=np.std(V)/np.mean(V) #離散係數
    post[i,1]=np.max(V)-np.min(V)  #極差
    post[i,2]=sts.skewness(V)  #偏度
    post[i,3]=sts.kurtosis(V)  #峰度
    #post 為1
    post[i,4]='1'

pre=np.zeros((15,5))
#pre
for i in range(15):
    path='data/pre/' + str(i+1) + '.csv'
    #將V前1萬5千筆讀出來並分析,skiprows為跳過第一個row
    V = pd.read_csv(path, skiprows=1)['V'][0:20004]
    pre[i,0]=np.std(V)/np.mean(V)  #離散係數
    pre[i,1]=np.max(V)-np.min(V)   #極差
    pre[i,2]=sts.skewness(V)   #偏度
    pre[i,3]=sts.kurtosis(V)   #峰度
    #pre 為0
    pre[i,4]='0'

#把post和pre整合
data=np.vstack((post,pre))
# #正規化
# from sklearn import preprocessing
# #將前四項特徵正規化
# for i in range(4):
#     data[:,i] = preprocessing.scale([data[:,i]])
#     # = preprocessing.normalize()
#     print(sum(data[:,i]))
# print(data)

#資料處理，y為目標
from sklearn.model_selection import train_test_split  
X = data[:,:3] 
y = data[:,4]  
#分割資料 取70%為訓練資料，30%為測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  

#用SVC分群， kernel用poly，support vecotr machine
from sklearn.svm import SVC
#kernel: rbf, sigmoid
svclassifier = SVC(kernel='poly', degree=4, gamma='scale')  
svclassifier.fit(X_train, y_train)

#使用X_train測試效能
#y_pred是模型輸出
y_pred = svclassifier.predict(X_train)  
from sklearn.metrics import classification_report, confusion_matrix  
print('train confusion matrix:')
print(confusion_matrix(y_train, y_pred))
print('train classification report')
print(classification_report(y_train, y_pred))

#使用X_test測試效能
y_pred = svclassifier.predict(X_test)  
from sklearn.metrics import classification_report, confusion_matrix  
print('test confusion matrix:')
print(confusion_matrix(y_test, y_pred))  
print('test classification report')
print(classification_report(y_test, y_pred))
