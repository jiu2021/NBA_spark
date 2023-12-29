import os
import json

def isExist(filename):
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 拼接文件路径
    file_path = os.path.join(current_directory, filename)
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 打开文件并读取内容
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # 对文件内容进行处理
        json_data = json.loads(file_content)
        return json_data
    else:
        print("文件不存在")
        return []
    
# isExist('clusters.json')