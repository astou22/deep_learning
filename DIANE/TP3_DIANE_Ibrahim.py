  
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import datasets  
from minisom import MiniSom
import numpy as np
import sys

iris = datasets.load_iris()
X = iris.data
Y = iris.target

som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5, neighborhood_function='triangle', random_seed=10) 
som.train_random(X, 100)
q_error=som.quantization_error(X)
t_error=som.topographic_error(Y)
print("erreur topologiue: ", q_error)
print("erreur de quantification: ", t_error)

Xn = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)
som = MiniSom(7, 7, 4, sigma=3, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)
som.train_random(Xn, 100)
q_error=som.quantization_error(Xn)
t_error=som.topographic_error(Xn)
print("erreur topologiue: ", q_error)
print("erreur de quantification: ", t_error)

plt.figure(figsize=(7, 7))

plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
r = np.zeros(len(target), dtype=int)
r[Y == 'Iris-setosa'] = 0
r[Y == 'Iris-versicolor'] = 1
r[Y == 'Iris-virginica'] = 2

 # utilisation de differente couleur pour marquer les labels
markers = ['o', '.', '+']
colors = ['C0', 'C1', 'C2']
for cpt, xx in enumerate(X):
    w = som.winner(xx)  
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cpt]], markerfacecolor='None',
                markeredgecolor=colors[t[cpt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.savefig('iris_som.png')
plt.show()

labels_map = som.labels_map(X, Y)
label_names = np.unique(Y)

plt.figure(figsize=(7, 7))
the_grid = GridSpec(7, 7)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.legend(patches, label_names, bbox_to_anchor=(0, 1.5), ncol=3)
plt.savefig('som_iris.png')
plt.show()

som = MiniSom(7, 7, 4, sigma=3, learning_rate=0.5,
              neighborhood_function='triangle', random_seed=10)

#2
matrice_U=som._distance_from_weights(X)
matrice_D=som.distance_map()
print("Matrix_U: \n",matrice_U)
print("Matrix_D : \n",matrice_D)

Xn = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, X)
som.pca_weights_init(Xn)
som.train_batch(Xn, 1000, verbose=True)  


plt.figure(figsize=(7, 7))

# affichage des distances
plt.pcolor(som.distance_map().T, cmap='bone_r')
plt.colorbar()
t = np.zeros(len(target), dtype=int)
r[Y == 'Iris-setosa'] = 0
r[Y == 'Iris-versicolor'] = 1
r[Y == 'Iris-virginica'] = 2


 # utilisation de differente couleur pour marquer les labels
markers = ['o', '.', '+']
colors = ['C0', 'C1', 'C2']
for cpt, xx in enumerate(X):
    w = som.winner(xx) 
    # palce a marker on the winning position for the sample xx
    plt.plot(w[0]+.5, w[1]+.5, markers[t[cpt]], markerfacecolor='None',
                markeredgecolor=colors[t[cpt]], markersize=12, markeredgewidth=2)
plt.axis([0, 7, 0, 7])
plt.savefig('iris_som.png')
plt.show()

labels_map = som.labels_map(X, Y)
label_names = np.unique(Y)

plt.figure(figsize=(7, 7))
the_grid = GridSpec(7, 7)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in label_names]
    plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)
plt.legend(patches, label_names, bbox_to_anchor=(0, 1.5), ncol=3)
plt.savefig('som_iris.png')
plt.show()

iris = datasets.load_iris()
X = iris.data
y = iris.target

model=KMeans(n_clusters=3)
model.fit(X)

plt.scatter(X[:,1], X[:,2], c=y, s=40)
plt.title('Data_iris')
plt.show()

#Visualisation des clusters formés par K-Means
plt.scatter(X[:, 1], X[:, 2], c=model.labels_, s=40)
plt.title('K-means ')
plt.show()

def plot_dendrogram(model, **kwargs):
    
    compteur = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 
            else:
                current_count += compteur[child_idx - n_samples]
        compteur[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, compteur]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(X)

plot_dendrogram(model, truncate_mode='level', p=3)
plt.show()
