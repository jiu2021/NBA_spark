import json
import tqdm
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, collect_list


class Model:
    def __init__(self, path: str) -> None:
        spark = SparkSession.builder.getOrCreate()
        # Load data
        df = spark.read.csv(path, header = True)
        df = self.preprocess(df)
        # PCA
        self.determine_pca(df)
        df = self.pca.transform(df)
        # Train model
        # self.determine_cluster(df)
        self.model = KMeans(featuresCol='pca_features', k=10).fit(df)
        results = self.model.transform(df).select(['player_name', 'prediction'])
        results = results.groupBy('prediction').agg(collect_list('player_name').alias('player_name'))
        self.prediction_dict = results.rdd.collectAsMap()
        with open("./clusters.json", "w") as fp:
            json.dump(self.prediction_dict, fp)

    def __call__(self, df) -> str:
        df = self.preprocess(df)
        df = self.pca.transform(df)
        df = self.model.transform(df).select(['player_name', 'prediction'])
        results = {}
        for row in df.collect():
            results[row['player_name']] = self.prediction_dict[row["prediction"]]
        return json.dumps(results)

    @staticmethod
    def preprocess(df):
        column_to_keep = ['player_name', 'player_height', 'player_weight', 'pts', 'reb', 'ast', 'net_rating', 'oreb_pct', 'dreb_pct', 'usg_pct', 'ts_pct', 'ast_pct']
        # Drop useless columns
        df = df.select(column_to_keep)
        column_to_keep.pop(0)
        # Average players' performance
        df = df.groupBy('player_name').agg(*[avg(column).alias(column) for column in column_to_keep])
        # Convert to feature vectors
        assemble = VectorAssembler(inputCols=column_to_keep, outputCol='features')
        df = assemble.transform(df)
        # Normalize
        scale = StandardScaler(inputCol='features', outputCol='standardized')
        scale = scale.fit(df)
        df = scale.transform(df)
        return df

    def determine_pca(self, df):
        pca = PCA(k=11, inputCol='standardized', outputCol='pca_features')
        pca = pca.fit(df)
        var_ratio = pca.explainedVariance.toArray()
        cum_var_ratio = [sum(var_ratio[:i+1]) for i in range(11)]
        fig, ax = plt.subplots(1,1)
        ax.plot(range(1, 12), cum_var_ratio)
        ax.set_xlabel('k')
        ax.set_ylabel('Cumulative Variance Ratio')
        fig.savefig("./img/pca.png")
        k = min(i + 1 for i in range(11) if cum_var_ratio[i] > 0.9)
        print(f"Chosen k_pca = {k}")
        pca = PCA(k=k, inputCol='standardized', outputCol='pca_features')
        self.pca = pca.fit(df)

    def determine_cluster(self, df):
        df = self.preprocess(df)
        df = self.pca.transform(df)
        scores = []
        evaluator = ClusteringEvaluator(featuresCol='standardized')
        kmin, kmax = 2, 51
        scores = []
        for i in tqdm.tqdm(range(kmin, kmax)):
            model = KMeans(featuresCol='pca_features', k=i).fit(df)
            results = model.transform(df)
            score = evaluator.evaluate(results)
            scores.append(score)
        fig, ax = plt.subplots(1,1)
        ax.plot(range(kmin, kmax), scores)
        ax.set_xlabel('k')
        ax.set_ylabel('cost')
        fig.savefig("./img/kmeans.png")


if __name__ == "__main__":
    model = Model("nba_all_seasons.csv")
    spark = SparkSession.builder.getOrCreate()
    # Load data
    df = spark.read.csv("nba_all_seasons.csv", header = True)
    print(model(df.limit(2)))
