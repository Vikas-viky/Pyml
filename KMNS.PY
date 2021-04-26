#Import libraries 
import pandas as pd 
from sklearn.cluster import KMeans 
#import the dataset 
df = pd.read_csv('iris.csv') 
print(df.head(10)) 
x = df.iloc[:, [0,1,2,3]].values # Load Input Attributes 
kmeans5 = KMeans(n_clusters=5) # Create 5 Clusters 
y_kmeans5 = kmeans5.fit_predict(x) 
print(y_kmeans5) #Prints the clusters for each record 
print(kmeans5.cluster_centers_) 
Error =[] 
for i in range(1, 11): 
 kmeans = KMeans(n_clusters = i).fit(x) 
 kmeans.fit(x) 
 Error.append(kmeans.inertia_) 
import matplotlib.pyplot as plt 
plt.plot(range(1, 11), Error) 
plt.title('Elbow method') 
plt.xlabel('No of clusters') 
plt.ylabel('Error') 
plt.show() 