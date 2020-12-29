from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
import numpy as np #para manipular os vetores
import pandas as pd #para abrir arquivos
from sklearn.preprocessing import MinMaxScaler #para normalizar

arq = pd.read_csv('dataset.csv', sep=';')  
dataset = np.array(arq)

#normalizando array
#normalized = MinMaxScaler(feature_range = (50, 200)) #valor para normalizacao
#x_norm = normalized.fit_transform(dataset) #normalizando

# define the model
model = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat = model.fit_predict(dataset)



# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(dataset[row_ix, 0], dataset[row_ix, 1])
# show the plot
pyplot.show()