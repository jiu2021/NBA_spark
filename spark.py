from pyspark import SparkContext
from pyspark.sql import SparkSession
import json


class SparkServer:
    def __init__(self, filename):
        self.sc = SparkContext( 'local', 'test')
        self.sc.setLogLevel("WARN")
        self.spark = SparkSession.builder.getOrCreate()
        self.df = self.spark.read.csv(filename, header = True)  #dataframe
        self.playername = ''
        self.playerdata = []

    def getRawPlayer(self, playername):
        raw_data = self.df.filter(self.df['player_name'] == playername).collect()
        json_list = [row.asDict() for row in raw_data]
        # json_data_str = json.dumps(json_data)
        return json_list
    
#统计各类型的专辑数量（只显示总数量大于2000的十种专辑类型）
def genre(sc, spark, df):
    #按照genre字段统计每个类型的专辑总数，过滤出其中数量大于2000的记录
    #并取出10种类型用于显示
    j = df.groupBy('genre').count().filter('count > 2000').take(10)
    #把list数据转换成json字符串，并写入到static/data目录下的json文件中
    f = open('static/data/genre.json', 'w')
    f.write(json.dumps(j))
    f.close()

#统计各个类型专辑的销量总数
def genreSales(sc, spark, df):
    j = df.select('genre', 'num_of_sales').rdd\
        .map(lambda v: (v.genre, int(v.num_of_sales)))\
            .reduceByKey(lambda x, y: x + y).collect()
    f = open('static/data/genre-sales.json', 'w')
    f.write(json.dumps(j))
    f.close()

#统计每年发行的专辑数量和单曲数量
def yearTracksAndSales(sc, spark, df):
    #把相同年份的专辑数和单曲数量相加，并按照年份排序
    result = df.select('year_of_pub', 'num_of_tracks').rdd\
        .map(lambda v: (int(v.year_of_pub), [int(v.num_of_tracks), 1]))\
            .reduceByKey(lambda x, y: [x[0] + y[0], x[1] + y[1]])\
                .sortByKey()\
                .collect()

    #为了方便可视化实现，将列表中的每一个字段分别存储
    ans = {}
    ans['years'] = list(map(lambda v: v[0], result))
    ans['tracks'] = list(map(lambda v: v[1][0], result))
    ans['albums'] = list(map(lambda v: v[1][1], result))
    f = open('static/data/year-tracks-and-sales.json', 'w')
    f.write(json.dumps(ans))
    f.close()


#取出总销量排名前五的专辑类型
def GenreList(sc, spark, df):
    genre_list = df.groupBy('genre').count()\
        .orderBy('count',ascending = False).rdd.map(lambda v: v.genre).take(5)
    return genre_list
    
    
#分析总销量前五的类型的专辑各年份销量
def GenreYearSales(sc, spark, df, genre_list):
    #过滤出类型为总销量前五的专辑，将相同类型、相同年份的专辑的销量相加，并进行排序。
    result = df.select('genre', 'year_of_pub', 'num_of_sales').rdd\
        .filter(lambda v: v.genre in genre_list)\
            .map(lambda v: ((v.genre, int(v.year_of_pub)), int(v.num_of_sales)))\
                .reduceByKey(lambda x, y: x + y)\
                    .sortByKey().collect()

    #为了方便可视化数据提取，将数据存储为适配可视化的格式
    result = list(map(lambda v: [v[0][0], v[0][1], v[1]], result))
    ans = {}
    for genre in genre_list:
        ans[genre] = list(filter(lambda v: v[0] == genre, result))
    f = open('static/data/genre-year-sales.json', 'w')
    f.write(json.dumps(ans))
    f.close()


#总销量前五的专辑类型，在不同评分体系中的平均评分
def GenreCritic(sc, spark, df, genre_list):
    #过滤出类型为总销量前五的专辑，将同样类型的专辑的滚石评分、mtv评分，音乐达人评分分别取平均
    result = df.select('genre', 'rolling_stone_critic', 'mtv_critic', 'music_maniac_critic').rdd\
        .filter(lambda v: v.genre in genre_list)\
        .map(lambda v: (v.genre, (float(v.rolling_stone_critic), float(v.mtv_critic), float(v.music_maniac_critic), 1)))\
        .reduceByKey(lambda x, y : (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))\
        .map(lambda v: (v[0], v[1][0]/v[1][3], v[1][1]/v[1][3], v[1][2]/v[1][3])).collect()

    f = open('static/data/genre-critic.json', 'w')
    f.write(json.dumps(result))
    f.close()
    
    
#代码入口
if __name__ == "__main__":
    sc = SparkContext( 'local', 'test')
    sc.setLogLevel("WARN")
    spark = SparkSession.builder.getOrCreate()
    file = "albums.csv"
    df = spark.read.csv(file, header = True)  #dataframe

    genre_list = GenreList(sc, spark, df)

    genre(sc, spark, df)
    genreSales(sc, spark, df)
    yearTracksAndSales(sc, spark, df)
    GenreYearSales(sc, spark, df, genre_list)
    GenreCritic(sc, spark, df, genre_list)
