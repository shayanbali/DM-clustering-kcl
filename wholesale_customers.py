# Part 2: Cluster Analysis

# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import itertools
import matplotlib.pyplot as plt
def read_csv_2(data_file):
	df = pd.read_csv(data_file)
	df = df.drop(['channel', 'region'], axis=1)
	return df

# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
	df = df.select_dtypes(include=['number'])
	df = df.aggregate(['mean', 'std', 'min', 'max']).T
	return df

# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.
def standardize(df):
	df = df.select_dtypes(include=['number'])
	df = (df - df.mean()) / df.std()
	return df

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.
# To see the impact of the random initialization,
# using only one set of initial centroids in the kmeans run.
def kmeans(df, k):
	df = df.select_dtypes(include=['number'])
	kmeans = KMeans(n_clusters=k, init='random', n_init=1, random_state=0)
	kmeans.fit(df)
	return pd.Series(kmeans.labels_, index=df.index, name="Cluster")

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
	df = df.select_dtypes(include=['number'])
	kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
	kmeans.fit(df)
	return pd.Series(kmeans.labels_, index=df.index, name="Cluster")

# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
	df = df.select_dtypes(include=['number'])
	agg = AgglomerativeClustering(n_clusters=k)
	agg.fit(df)
	return pd.Series(agg.labels_, index=df.index, name="Cluster")

# Given a data set X and an assignment to clusters y
# return the Silhouette score of this set of clusters.
def clustering_score(X,y):
	score = silhouette_score(X, y)  

# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
	result = []
	list_of_algorithms = ['Kmeans', 'Agglomerative']
	list_of_data_types = ['Original', 'Standardized']
	list_of_k = [3, 5, 10]
	for algorithm in list_of_algorithms:
		for data_type in list_of_data_types:
			if data_type == 'Standardized':
				df = standardize(df)
			for k in list_of_k:
				if algorithm == 'Kmeans':
					y = kmeans(df, k)
				else:
					y = agglomerative(df, k)
					score = silhouette_score(df, y)
					result.append({'Algorithm': algorithm, 'data type': data_type, 'k': k, 'Silhouette Score': score})
	return pd.DataFrame(result)
		

# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
	return rdf['Silhouette Score'].max()

# Run the Kmeans algorithm with k=3 by using the standardized data set.
# Generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.
def scatter_plots(df):
	df = standardize(df)
	k = 3
	kmeans = KMeans(n_clusters=k, init='random', n_init=1, random_state=0)
	kmeans.fit(df)
	labels = kmeans.labels_
	attributes = df.columns
	pairs = list(itertools.combinations(attributes, 2))
	dir = 'cluster_results/'
	for pair in pairs:
		df.plot.scatter(x=pair[0], y=pair[1], c=labels, colormap='viridis')
		plt.show()
		file_path = f"{dir}scatter_{pair[0]}_add_{pair[1]}_k{k}.png"
		plt.savefig(file_path)
		plt.close()
	

