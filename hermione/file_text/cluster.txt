from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Cluster:
            
    @classmethod
    def analyzeK(cls, X, k_min = 2, k_max = 20):
        """
        Plot the result of the methods (elbow, silhouette and calinski_harabas) to find the best k
        
        Parameters
        ----------    
        X : array
            values ​​that will be used to find the best k
        k_min : int 
            minimum interval for K
        k_max : int
            maximum range for K
             
        Returns
        -------
        None
        """
        
        if X is None:
            raise Exception("Error: X is None.")
        if k_min is None or k_max is  None:
            raise Exception("Error: Range is None.")
        if k_min < 2:
            raise Exception("Error: k_min < 2")
        
        wss = []
        s_gmm = []
        s_kmeans = []
        ch_gmm = []
        ch_kmeans = []
        
        K = range(k_min, k_max)
                
        for k in K:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            gmm = GaussianMixture(n_components=k, covariance_type='full')
            gmm.fit(X)
            
            labels_kmeans = kmeans.predict(X)
            labels_gmm = gmm.predict(X)
            
            s_kmeans.append(metrics.silhouette_score(X, labels_kmeans, metric='euclidean'))
            s_gmm.append(metrics.silhouette_score(X, labels_gmm, metric='euclidean'))
            
            ch_kmeans.append(metrics.calinski_harabasz_score(X, labels_kmeans))
            ch_gmm.append(metrics.calinski_harabasz_score(X, labels_gmm))
                        
            wss.append(kmeans.inertia_)
                    
        cls._elbow(K, wss)
        cls._silhouette_coefficient(K, s_kmeans, s_gmm)
        cls._calinski_harabaz(K, ch_kmeans, ch_gmm)
        
    @classmethod
    def _elbow(cls, K, wss):
        """
        Function plots the result of the elbow method
        
        Parameters
        ----------    
        k : array
            possible k values
        k_min : array 
            Total WSS measures cluster compression and we want it to be as small as possible
        Returns
        -------
        None
        """
        plt.plot(K, wss, 'bx-')
        plt.xlabel('k')
        plt.ylabel('WSS')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()
       
    @classmethod
    def _silhouette_coefficient(cls, K, s_kmeans, s_gmm):
        """
        Function plots the result of the silhouette method for kmeans and Gaussian Mixture Models
        
        Parameters
        ----------    
        k : array
             k values
        s_kmeans : array 
             Silhouette kmeans values
        s_gmm : array 
            Silhouette Gaussian Mixture Models values
        
        Returns
        ----
        None
        """
        plt.plot(K, s_kmeans, 'xr-') # plotting t, a separately 
        plt.plot(K, s_gmm, 'ob-')
        plt.legend(["kmeans", "gmm"])
        plt.xlabel('k')
        plt.ylabel('Mean Silhouette Coefficient')
        plt.title('Mean Silhouette Coefficient for each k')
        plt.show()
        
    @classmethod
    def _calinski_harabaz(cls, K, ch_kmeans, ch_gmm):
        """
        Function plots the result of the calinski_harabaz method for kmeans and Gaussian Mixture Models

        Parameters
        ----------    
        k : array
            possible k values
        s_kmeans : array 
             calinski_harabaz kmeans values
        s_gmm : array 
             Gaussian Mixture Models values
         
        Returns
        -------
        None
        """
        plt.plot(K, ch_kmeans, 'xr-') # plotting t, a separately 
        plt.plot(K, ch_gmm, 'ob-')
        plt.legend(["kmeans", "gmm"])
        plt.xlabel('k')
        plt.ylabel('Calinski and Harabaz score')
        plt.title('Calinski and Harabaz score for each k')
        plt.show()
        
    @classmethod
    def plot_cluster(cls, df_res_algorithm, algorithm_name = "K-means"):
        """
        Function that plots clusters
    
        Parameters
        ----------    
        df_res_algoritmo : pd.DataFrame
            Dataframe must have the following columns (x, y, cluster)
        algorithm_name : str 
            algorithm name             
        Return
        -------
        None
        """      
        # verifica quantos clusters tem 
        qtde_cluster = df_res_algorithm.cluster.max()+1        
        plots = []        
        for cluster in range(qtde_cluster):
            p = plt.scatter(df_res_algorithm[df_res_algorithm['cluster'] == cluster].x, 
                            df_res_algorithm[df_res_algorithm['cluster'] == cluster].y)
            plots.append(p)
        plt.legend(tuple(plots),
               (tuple(["Cluster {}".format(c) for c in range(1, qtde_cluster+1)])), 
               loc=2, fontsize=8, bbox_to_anchor=(1.05, 1))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Clusters created by "+algorithm_name)
        plt.show()