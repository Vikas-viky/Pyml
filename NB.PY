from sklearn import datasets 
iris = datasets.load_iris() 
print("Features: ", iris.feature_names) 
print("Labels: ", iris.target_names) 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, 
random_state=109) 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB() 
gnb.fit(x_train, y_train) 
y_pred = gnb.predict(x_test) 
from sklearn import metrics 
print("Accuracy: " ,metrics.accuracy_score(y_test, y_pred))