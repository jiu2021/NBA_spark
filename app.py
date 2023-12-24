# coding:utf-8
from flask import render_template
from flask import Flask, request
from spark import SparkServer
import json
app = Flask(__name__)

# 主页
@app.route('/')
def index():
    # 使用 render_template() 方法来渲染模板
    return render_template('index.html')

# 渲染html模板页
@app.route('/<filename>')
def req_file(filename):
    return render_template(filename)

# 搜索球员
@app.route('/getPlayer', methods=['POST'])
def getPlayer():
    # 从请求中获取输入参数
    input_value = request.form.get('inputValue')
    
    # 在控制台打印输入参数
    print(input_value)
    
    # 返回json数据
    res = mySpark.getRawPlayer(input_value)
    if len(res) != 0:
        mySpark.playername = input_value
        mySpark.playerdata = res
    print(mySpark.playername)
    return res

# 搜索球员
@app.route('/getCurPlayer', methods=['GET'])
def getCurPlayer():
    # 返回json数据
    return [mySpark.playername, mySpark.playerdata]

if __name__ == '__main__':   
    app.DEBUG=True#代码调试立即生效
    app.jinja_env.auto_reload = True#模板调试立即生效
    mySpark = SparkServer('nba_all_seasons.csv')
    app.run()#用 run() 函数来让应用运行在本地服务器上

