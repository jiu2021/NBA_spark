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
    
def filterStar(star_arr = []):
    data = isExist('clusters.json')
    res = []
    if len(data) != 0:
        for n in range(0, 10):
            tmp = []
            for star in star_arr:
                if star in data[str(n)]:
                    tmp.append(star)
            res.append(tmp)

    print(res)

players = [
    "Joel Embiid",
    "Luka Doncic",
    "LeBron James",
    "Giannis Antetokounmpo",
    "Karl-Anthony Towns",
    "Anthony Davis",
    "Nikola Jokic",
    "Kevin Durant",
    "Shaquille O'Neal",
    "Karl Malone",
    "Tim Duncan",
    "Zion Williamson",
    "Russell Westbrook",
    "Michael Jordan",
    "James Harden",
    "Charles Barkley",
    "DeMarcus Cousins",
    "Jayson Tatum",
    "Chris Webber",
    "Stephen Curry",
    "Damian Lillard",
    "Trae Young",
    "Kobe Bryant",
    "Kevin Garnett",
    "Donovan Mitchell",
    "Allen Iverson",
    "Ja Morant",
    "Chris Bosh",
    "Dwight Howard",
    "Yao Ming"
]

# filterStar(players)

[
    [],
    [],
    [],
    [],
    [],
    [
        'Karl-Anthony Towns',
        'Anthony Davis',
        "Shaquille O'Neal",
        'Tim Duncan',
        'Charles Barkley',
        'DeMarcus Cousins',
        'Kevin Garnett',
        'Chris Bosh',
        'Dwight Howard',
        'Yao Ming'
    ],
    [
        'Joel Embiid',
        'Luka Doncic',
        'LeBron James',
        'Giannis Antetokounmpo',
        'Nikola Jokic',
        'Kevin Durant',
        'Karl Malone',
        'Zion Williamson',
        'Russell Westbrook',
        'Michael Jordan',
        'James Harden',
        'Jayson Tatum',
        'Chris Webber',
        'Stephen Curry',
        'Damian Lillard',
        'Trae Young',
        'Kobe Bryant',
        'Donovan Mitchell',
        'Allen Iverson',
        'Ja Morant'
    ],
    [],
    [],
    []
]
