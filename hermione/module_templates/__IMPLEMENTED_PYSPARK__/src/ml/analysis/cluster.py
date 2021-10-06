from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns

class SparkCluster:
            
    @classmethod
    def analyzeK(cls, df, k_min = 2, k_max = 20):
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
        
        if k_min is None or k_max is  None:
            raise Exception("Error: Range is None.")
        if k_min < 2:
            raise Exception("Error: k_min < 2")
        
        wss = []
        s_gmm = []
        s_kmeans = []
        
        K = range(k_min, k_max)
                
        for k in K:
            kmeans = KMeans(k=k).fit(df)
            gmm = GaussianMixture(k=k).fit(df)
            evaluator = ClusteringEvaluator()
            
            labels_kmeans = kmeans.transform(df)
            labels_gmm = gmm.transform(df)
            
            s_kmeans.append(evaluator.evaluate(labels_kmeans))
            s_gmm.append(evaluator.evaluate(labels_gmm))
            wss.append(kmeans.summary.trainingCost)
                    
        cls.__elbow(K, wss)
        cls.__silhouette_coefficient(K, s_kmeans, s_gmm)
        
    @classmethod
    def __elbow(cls, K, wss):
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
    def __silhouette_coefficient(cls, K, s_kmeans, s_gmm):
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
    def plot_cluster(cls, df_res_algorithm, x, y, labels):
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
        sns.scatterplot(x =x, y = y, hue=labels, data = df_res_algorithm)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Clusters")
        plt.show()