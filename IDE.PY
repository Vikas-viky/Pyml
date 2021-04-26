from sklearn.tree import DecisionTreeClassifier 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
dataset=pd.read_csv('pima-indians-Diabetes.csv') 
print(dataset.head()) 
train_features=dataset.iloc[:80,:-1] 
test_features=dataset.iloc[80:,:-1] 
train_targets=dataset.iloc[:80,-1] 
test_targets=dataset.iloc[80:,-1] 
tree1=DecisionTreeClassifier(criterion='entropy').fit(train_features,train_targets) 
prediction=tree1.predict(test_features) 
cm=confusion_matrix(test_targets,prediction) 
print('Confusion Matrix:\n',cm) 
TP=cm[0][0] 
FP=cm[0][1] 
FN=cm[1][0] 
TN=cm[1][1] 
print('False Positive \n {}'.format(FP)) 
print('False Negative \n {}'.format(FN)) 
print('True Positive \n {}'.format(TP)) 
print('True Negative \n {}'.format(TN)) 
TPR=TP/(TP+FN) 
print('Senstivity \n {}'.format(TPR)) 
TNR=TN/(TN+FP) 
print('Specificity \n {}'.format(TNR)) 
Precision=TP/(TP+FP) 
print('Precision \n {}'.format(Precision))
Recall=TP/(TP+FN) 
print('Recall \n {}'.format(Recall)) 
Acc=(TP+TN)/(TP+TN+FP+FN) 
print('Accuracy \n {}'.format(Acc)) 
Fscore=2*(Precision*Recall)/(Precision+Recall) 
print('FScore \n {}'.format(Fscore))