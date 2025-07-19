import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# loading data from CSV
iris = pd.read_csv(r'D:/Usman/Data Science/Iris.csv')

dimensions = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                   'PetalWidthCm']]

# printing dataframe's first 5 rows to check df structure and scaling features
print(iris.head())
X = scale(dimensions)

# finding optimum number of clusters by training K-Means for cluster = 1 to 10
cost = []
for i in range(1, 11):
    clustering = KMeans(n_clusters=i, n_init=10, random_state=5,
                        max_iter=500)
    y_km = clustering.fit_predict(X)
    cost.append(clustering.inertia_)

# plotting cluster number against the function cost
plt.plot(range(1, 11), cost, marker='o', c='red')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost Function')
plt.grid()
plt.show()

# calculating K-Means for the optimal cluster number i.e 3
clustering = KMeans(n_clusters=3, n_init=10, random_state=5,
                    max_iter=500)
y_km = clustering.fit_predict(X)

# plotting clusters according to sepal dimensions
plt.subplot()
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='blue',
            marker='v', edgecolors='black', label='Cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='green',
            marker='s', edgecolors='black', label='Cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='gray',
            marker='o', edgecolors='black', label='Cluster 3')
plt.title("Clusters based on sepal length and width")
# plotting cluster centroids
plt.scatter(clustering.cluster_centers_[:,0], clustering.cluster_centers_[:,1],
            marker='x', s=100, c='red', label='Centroids')
plt.legend()
plt.grid()
plt.show()

# plotting clusters according to petal dimensions
plt.scatter(X[y_km == 0, 2], X[y_km == 0, 3], s=50, c='blue',
            marker='v', edgecolors='black', label='Cluster 1')
plt.scatter(X[y_km == 1, 2], X[y_km == 1, 3], s=50, c='green',
            marker='s', edgecolors='black', label='Cluster 2')
plt.scatter(X[y_km == 2, 2], X[y_km == 2, 3], s=50, c='gray',
            marker='o', edgecolors='black', label='Cluster 3')
plt.title("Clusters based on petal length and width")
# plotting cluster centroids
plt.scatter(clustering.cluster_centers_[:,2], clustering.cluster_centers_[:,3],
            marker='x', s=100, c='red', label='Centroids')
plt.legend()
plt.grid()
plt.show()
