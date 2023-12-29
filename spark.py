from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import abs
from rank import Rank
from relate import Relate
from utils import isExist
import json


class SparkServer:
    def __init__(self, filename):
        self.sc = SparkContext( 'local', 'test')
        self.sc.setLogLevel("WARN")
        self.spark = SparkSession.builder.getOrCreate()
        self.df = self.spark.read.csv(filename, header = True)  #dataframe
        self.playername = ''
        self.playerdata = []
        self.score_df = None

    def getRawPlayer(self, playername):
        raw_data = self.df.filter(self.df['player_name'] == playername).collect()
        json_list = [row.asDict() for row in raw_data]
        return json_list
    
    def getRank(self):
        rank = Rank(self.df, self.spark)
        self.score_df = rank.getDfWithRank()
        json_list = [row.asDict() for row in self.score_df.collect()]
        return json_list
    
    def getRelate(self, playername):
        row = self.score_df.filter(self.score_df['player_name'] == playername).first()
        target_score = row.score
        
        # 如果有聚类结果，直接读json文件
        data = isExist('clusters.json')
        if len(data) != 0:
            for n in range(0, 10):
                if playername in data[str(n)]:
                    res_list = data[str(n)]
                    break
        else:
            relate = Relate(self.df)
            player_df = self.df.filter(self.df['player_name'] == playername)
            res = relate(player_df)
            res_list = json.loads(res)[playername]
        
        
        scale = 0
        if(target_score > 0.5):
            scale = 0.04
        elif(target_score > 0.4):
            scale = 0.01
        else:
            scale = 0.001

        # 过滤出综合能力相似的球员
        maybe_data = self.score_df.filter(abs(self.score_df.score - target_score) < scale).collect()
        maybe_list = [row.asDict() for row in maybe_data]
        for index, maybe in enumerate(maybe_list):
            if maybe in res_list:
                continue
            else:
                del maybe_list[index]

        print(target_score)
        return maybe_list
        
    def getRandom(self):
        player = self.df.sample(0.01).limit(1)
        playername = player.select("player_name").first()[0]
        return playername