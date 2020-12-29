#Affinity Propagation para clusterização e comparação com o k-means

from numpy import unique
from numpy import where

from sklearn.cluster import AffinityPropagation

from matplotlib import pyplot

import numpy as np #para manipular os vetores

import pandas as pd #para abrir arquivos
from matplotlib import pyplot as plt #para plotar os gráficos

from sklearn.preprocessing import MinMaxScaler #para normalizar


#carregando arquivo
arq = pd.read_csv('dataset.csv', sep=';')  

#criando array do arquivo csv
dataset = np.array(arq)

#normalizando array
#normalized = MinMaxScaler(feature_range = (50, 200)) #valor para normalizacao
#x_norm = normalized.fit_transform(dataset) #normalizando

#parâmetros dpara o eaeans
model = AffinityPropagation(damping=0.9)
model.fit(dataset)

#passando imagem (HSV) para predicao
imgjpg = np.array([[128,152,122, 0.156387, 3.86, 0.526982]])

#predizando os dados para o dataset passado
pred_y = model.predict(dataset)

#imprime a predição da imagem passada
print(model.predict(imgjpg))

# assign a cluster to each example
yhat = model.predict(dataset)
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
