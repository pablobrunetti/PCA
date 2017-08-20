import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Leitura dos dados no CSV
df = pd.read_csv(
    filepath_or_buffer='dataset/iris.data',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()

#Armazenamento dos dados em uma lista em forma de matriz
X = df.ix[:,0:4].values
y = df.ix[:,4].values

#Matriz de covariancia
mean_vec = np.mean(X, axis=0)

for i in range(len(X)):
    X[i,:] = X[i,:] - mean_vec[:]

cov_mat = np.cov((X[:,0],X[:,1],X[:,2],X[:,3]))



#cov_mat = np.cov(X.T)
#cov_mat = ((X - mean_vec).T.dot((X - mean_vec))) / (X.shape[0]-1)
print('Matriz de Covariancia \n%s' %cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Autovetores \n%s' %eig_vecs)
print('\nAutovalores \n%s' %eig_vals)


# Fazer uma lista de tuplas (eigenvalue,
eig_pairs = [(eig_vals[i], eig_vecs[:,i]) for i in range(len(eig_vals))]

# Ordenar as tuplas do maior para menor(eigenvalue, eigenvector)
eig_pairs.sort()
eig_pairs.reverse()

# Confirmar visualmente que a lista esta corretamenta ordenada por ordem decrescente

#print('Eigenvalues in descending order:')
#for i in eig_pairs:
#    print(i[0])

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

#print('Matrix W:\n', matrix_w)
#
label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}

Y = X.dot(matrix_w)

assert Y.shape == (150,2), "A matriz nao possui dimensao 150x2."

plt.plot(Y[0:49,1],Y[0:49,0], '^', markersize=7, color='red', alpha=0.5, label='Virginia')
plt.plot(Y[50:99,1],Y[50:99,0], 'o', markersize=7, color='blue', alpha=0.5, label='Versicolor')
plt.plot(Y[100:149,1],Y[100:149,0], 's', markersize=7, color='green', alpha=0.5, label='Setosa')

plt.xlim([-1.5,1.5])
plt.ylim([-4,4])

plt.xlabel('2 Componente Principal')
plt.ylabel('1 Componente Principal')
plt.legend()
plt.title('Dois componentes principais da Base de Dados Iris')

plt.tight_layout
plt.grid()
plt.show()