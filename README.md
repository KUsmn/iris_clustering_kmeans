# 🌸 Iris Clustering with K-Means

This project applies **K-Means Clustering** (unsupervised learning) to group
Iris flower samples into natural clusters based on sepal and petal dimensions.

Using **scikit-learn**, it scales features, determines the optimal number of
clusters using the **Elbow Method**, and visualizes the clusters with centroids
in both sepal and petal feature spaces.

---

## 📁 Project Structure

```
📁 iris_clustering_kmeans.py       – Complete K-Means clustering implementation  
📁 README.md                       – Overview and execution instructions  
```

---

## 🧠 Key Highlights

- Loads Iris dataset from CSV  
- Scales feature data using `sklearn.preprocessing.scale`  
- Uses **Elbow Method** to determine optimal cluster count  
- Applies **KMeans** from scikit-learn  
- Visualizes clusters based on both **sepal** and **petal** features  
- Plots final cluster centroids  

---

## 🔧 Installation

Install dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 🚀 Run the Script

1. Update the dataset path in the script if needed:
   ```python
   iris = pd.read_csv("path/to/Iris.csv")
   ```

2. Then run:

```bash
python iris_clustering_kmeans.py
```

---

## 📊 Output Overview

### Elbow Method Plot

Visualizes the cost function (inertia) for 1–10 clusters to identify the
"elbow" point (typically 3 for Iris dataset), which indicates the optimal
number of clusters.

### Cluster Visualizations

The script produces two scatter plots:
1. **Clusters based on Sepal Length vs Width**
2. **Clusters based on Petal Length vs Width**

Each cluster is color-coded and includes its centroid (red '×').

---

## 🌼 Dataset Info

The [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) contains
150 samples from three iris species:

- *Setosa*
- *Versicolor*
- *Virginica*

Each sample has four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

## 🔖 Tags

`#KMeans` `#Clustering` `#IrisDataset` `#UnsupervisedLearning`  
`#Python` `#ScikitLearn` `#DataScience` `#ElbowMethod` `#Visualization`
