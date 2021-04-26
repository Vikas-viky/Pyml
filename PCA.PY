# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60 
import pandas as pd 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data" 
# load dataset into Pandas DataFrame 
#data = pd.read_csv("data5.csv") 
#df = pd.read_csv("Iris.csv", names=['sepal length','sepal width','petal length','petal 
width','target']) 
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target']) 
print(df.head) 
from sklearn.preprocessing import StandardScaler 
features = ['sepal length', 'sepal width', 'petal length', 'petal width'] 
# Separating out the features 
x = df.loc[:, features].values 
# Separating out the target 
y = df.loc[:,['target']].values 
# Standardizing the features 
x = StandardScaler().fit_transform(x) 
print("Standardized features ", x) 
from sklearn.decomposition import PCA 
pca = PCA(n_components=2) 
principalComponents = pca.fit_transform(x) #Apply PCA 
principalDf = pd.DataFrame(data = principalComponents, 
columns = ['principal component 1', 'principal component 2']) 
finalDf = pd.concat([principalDf, df[['target']]], axis = 1) 
print(pca.explained_variance_ratio_) 