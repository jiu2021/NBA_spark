from pyspark.sql import SparkSession
from pyspark.sql import functions as F


class Rank:
    def __init__(self, df, spark):
        # dataframe
        self.df = df
        # 获取唯一球员
        self.players = [row['player_name'] for row in df.select('player_name').distinct().collect()]
        self.spark = spark

    def load_data(self):
        num_players = len(self.players)
        data = [[0.0] * 9 for _ in range(num_players)]
        dataset = [[] for _ in range(num_players)]

        for row in self.df.collect():
            name = row['player_name']
            index = self.players.index(name)
            temp = [
                float(row['pts']),
                float(row['reb']),
                float(row['ast']),
                float(row['net_rating']),
                float(row['oreb_pct']),
                float(row['dreb_pct']),
                float(row['usg_pct']),
                float(row['ts_pct']),
                float(row['ast_pct'])
            ]
            dataset[index].append(temp)

        for i in range(num_players):
            data[i] = [sum(x) / len(x) if len(x) > 0 else 0.0 for x in zip(*dataset[i])]

        col_norms = [sum(x ** 2 for x in col) ** 0.5 if len(col) > 0 else 0.0 for col in zip(*data)]
        data_normalized = [[x / col_norms[j] if col_norms[j] != 0.0 else 0.0 for j, x in enumerate(row)] for row in data]
        return data_normalized
    
    def grey_relation_analysis(self, data_normalized):
        max_arr = [max(row) for row in data_normalized]
        max_arr_column = [[x] for x in max_arr]
        # print(max_arr_column)
        # print(len(max_arr_column))
        results = [[abs(x - max_arr_column[i][0]) for x in row] for i, row in enumerate(data_normalized)]

        max_value = max(max(results))
        min_value = min(min(results))
        alpha = [0.9, 0.43, 0.015, 0.075, 0.015, 0.05, 0.2, 0.1, 0.075]
        roi = 0.48
        results = [
            [
                1 / (x + roi * max_value) * (min_value + roi * max_value) * alpha[j]
                for j, x in enumerate(row)
            ]
            for row in results
        ]
        r = [sum(x) for x in zip(*results)]
        r = [x / sum(r) for x in r]
        return r
    
    def topsis(self, data_normalized, r):
        max_elements = [max(row) for row in zip(*data_normalized)]
        min_elements = [min(row) for row in zip(*data_normalized)]

        max_result = [[(x - max_elements[j]) ** 2 for j, x in enumerate(row)] for row in data_normalized]
        min_result = [[(x - min_elements[j]) ** 2 for j, x in enumerate(row)] for row in data_normalized]

        score1 = [sum(x * r[j] for j, x in enumerate(row)) for row in max_result]
        score2 = [sum(x * r[j] for j, x in enumerate(row)) for row in min_result]

        score = [score2[i] ** 0.5 / (score1[i] ** 0.5 + score2[i] ** 0.5) for i, x in enumerate(score2)]
        # print(score)
        return score
    
    def getDfWithRank(self):
        # 加载数据并执行分析
        data_normalized = self.load_data()
        r = self.grey_relation_analysis(data_normalized)
        scores = self.topsis(data_normalized, r)

        # 创建 Spark DataFrame 来存储结果
        score_data = list(zip(self.players, scores))
        score_df = self.spark.createDataFrame(score_data, ['player_name', 'score'])
        score_df = score_df.sort("score", ascending=False)
        score_df.show()
        # 显示结果
        # score_df.show()
        return score_df