#Implementation of Kmeans from scratch and using sklearn
#Loading the required modules 
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
 
def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [0, 1, 2, 3, 4 ]
    colors = ['y', 'y', 'b', 'b', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'],
                   c = color, s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

def plot_samples(projected, labels, title):    
    fig = plt.figure()
    u_labels = np.unique(labels)
    for i in u_labels:
        plt.scatter(projected[labels == i , 0] , projected[labels == i , 1] , label = i,
                    edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('tab10', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend()
    plt.title(title)

 
def main():
    #Load dataset Digits
    input_file = 'hcvdatclearnumerico.csv'
    names = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
    features = ['Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
    target = 'Category'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    
    # Separating out the features
    x = df.loc[:, features].values

    # Separating out the target
    y = df.loc[:,[target]].values

    # # Standardizing the features
    # x = MinMaxScaler().fit_transform(x) #MIN-MAX
    # normalizedDf = pd.DataFrame(data = x, columns = features)
    # normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)
    x = StandardScaler().fit_transform(x) #ZSCORE
    normalizedDf = pd.DataFrame(data = x, columns = features)
    normalizedDf = pd.concat([normalizedDf, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalizedDf,"Dataframe Normalized")

    # PCA projection
    # pca = PCA(2)    
    # principalComponents = pca.fit_transform(x)
    # print("Explained variance per component:")
    # print(pca.explained_variance_ratio_.tolist())
    # print("\n\n")
    
    # principalDf = pd.DataFrame(data = principalComponents[:,0:2], 
    #                            columns = ['principal component 1', 
    #                                       'principal component 2'])
    # finalDf = pd.concat([principalDf, df[[target]]], axis = 1)    
    # ShowInformationDataFrame(finalDf,"Dataframe PCA")
    
    # VisualizePcaProjection(finalDf, target)
    
    #Transform the data using PCA
    pca = PCA(2)
    projected = pca.fit_transform(df.data)
    print(pca.explained_variance_ratio_)
    print(df.data.shape)
    print(projected.shape)    
    plot_samples(projected, df.target, 'Original Labels') 
 
    #Applying sklearn GMM function
    gm  = GaussianMixture(n_components=5).fit(principalDf)
    print(gm.weights_)
    print(gm.means_)
    x = gm.predict(principalDf)

    #Visualize the results sklearn
    plot_samples(principalDf, x, 'Clusters Labels GMM')

    plt.show()
 

if __name__ == "__main__":
    main()