from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import seaborn as sns


class SparkCluster:
    @classmethod
    def analyzeK(cls, df, algorithms, featuresCol="features", k_min=2, k_max=20):
        """
        Plot the result of the elbow and silhouette methods to find the best k number of clusters

        Parameters
        ----------
        df  : pyspark.sql.dataframe.DataFrame
            input Spark DataFrame to be analyzed

        algorithms : str or list of str
            Algorithms to be analyzed. Possible values are kmeans (KMeans) or gmm (Gaussian Mixture Models).

        featuresCol : str
            Column of type pyspark.ml.linalg.Vector that has the data to be used in the clusterization.

        k_min : int
            minimum interval for K

        k_max : int
            maximum range for K

        Returns
        -------
        Examples
        --------
        >>> from pyspark.ml.linalg import Vectors
        >>> data = [(1, Vectors.dense([0.0, 0.0])), (2, Vectors.dense([1.0, 54])),
        ...         (3, Vectors.dense([9.0, 27])), (4, Vectors.dense([8.0, 9.0]))]
        >>> df = spark.createDataFrame(data, ['id', "features"])
        >>> cluster = SparkCluster()
        >>> cluster.analyzeK(df, 'kmeans', k_max=5)
        >>> cluster.analyzeK(df, ['kmeans', 'gmm'], k_max=5)
        """

        if k_min is None or k_max is None:
            raise Exception("Error: Range is None.")
        if k_min < 2:
            raise Exception(
                "Error: the minimum number of clusters must be greater or equal to 2."
            )
        algorithms = algorithms if type(algorithms) is list else [algorithms]
        available_algo = {"kmeans": KMeans, "gmm": GaussianMixture}
        K = range(k_min, k_max + 1)
        evaluator = ClusteringEvaluator()
        results = dict()
        for algo in algorithms:
            model = available_algo[algo]
            results[algo] = dict()
            results[algo]["wss"] = []
            results[algo]["sil"] = []
            for k in K:
                cluster_model = model(k=k, featuresCol=featuresCol).fit(df)
                labels = cluster_model.transform(df)
                results[algo]["sil"].append(evaluator.evaluate(labels))
                if algo == "kmeans":
                    results[algo]["wss"].append(cluster_model.summary.trainingCost)
        for algo, res in results.items():
            if algo == "kmeans":
                cls.elbow_plot(K, res["wss"])
            cls.sil_plot(K, res["sil"], algo)

    @classmethod
    def elbow_plot(cls, K, wss):
        """
        Function that plots the result of the elbow method

        Parameters
        ----------
        K : Iterable[int]
            possible values of k

        wss : Iterable[float]
            The respective Within-Cluster-Sum of Squared Errors (WSS) for each value of k in `K`.

        Returns
        -------
        Examples
        --------
        >>> K = [2, 3, 4, 5]
        >>> wss = [650, 500, 300, 250]
        >>> cluster = SparkCluster()
        >>> cluster.elbow_plot(number_clusters, wss)
        """
        K = [str(k) for k in K]
        plt.plot(K, wss)
        plt.xlabel("k")
        plt.ylabel("WSS")
        plt.title("The Elbow Method showing the optimal k - KMEANS")
        plt.show()

    @classmethod
    def sil_plot(cls, K, sil, method):
        """
        Function that plots the result of the silhouette method

        Parameters
        ----------
        K : Iterable[int]
            possible values of k

        sil : Iterable[float]
            The respective average silhouette for each value of k in `K`.

        Returns
        -------
        Examples
        --------
        >>> number_clusters = [2, 3, 4, 5]
        >>> sil_scores = [0.4, 0.5, 0.55, 0.47]
        >>> cluster = SparkCluster()
        >>> cluster.sil_plot(number_clusters, sil_scores)
        """
        K = [str(k) for k in K]
        plt.plot(K, sil)  # plotting t, a separately
        plt.xlabel("k")
        plt.ylabel("Mean Silhouette Coefficient")
        plt.title(f"Mean Silhouette Coefficient for each k - {method.upper()}")
        plt.show()

    @classmethod
    def plot_cluster(cls, df_res_algorithm, x, y, labels):
        """
        Function that plots clusters

        Parameters
        ----------
        df_res_algoritmo : pd.DataFrame
            Pandas DataFrame to be plotted

        x : str
            Column to be plotted in the x-axis

        y : str
            Column to be plotted in the y-axis

        labels : str
            Column with the cluster labels

        Return
        -------
        Examples
        --------
        >>> data = [(1, 0.0, 0.0), (1, 1.0, 54.0), (1, 2.0, 30.0), (1, 0.5, 30.0),
        ...         (0, 9.0, 27.0), (0, 8.0, 9.0), (0, 6.5, 18.0), (0, 7.5, 35.0)]
        >>> df = spark.createDataFrame(data, ['cluster', "x", "y"]).toPandas()
        >>> cluster = SparkCluster()
        >>> cluster.plot_cluster(df, 'x', 'y', 'cluster')
        """
        # verifica quantos clusters tem
        sns.scatterplot(x=x, y=y, hue=labels, data=df_res_algorithm)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Clusters")
        plt.show()
