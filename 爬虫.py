import urllib.request
import pandas as pd
import numpy as np
import re
def askURL(url):
    head = {  # 模拟浏览器头部信息，向豆瓣服务器发送消息
        "User-Agent": "Mozilla / 5.0(Windows NT 10.0; Win64; x64) AppleWebKit / 537.36(KHTML, like Gecko) Chrome / 80.0.3987.122  Safari / 537.36"
    }
    request = urllib.request.Request(url,headers = head)
    response = urllib.request.urlopen(request)
    html = response.read(500000).decode("utf-8")
    return html
html = askURL("https://www.imdb.com/title/tt0113497/?ref_=ttcrv_ov")

links = np.array(pd.read_csv("links.csv"))
table = np.zeros((len(links),4),dtype = int)
table[:,0] = links[:,0]
findUR = re.compile(r'(\d+\.\d+K)</span><span class="label">User review|(\d+)</span><span class="label">User review|(\d+K)</span><span class="label">User review')
findCR = re.compile(r'(\d+\.\d+K)</span><span class="label">Critic review|(\d+)</span><span class="label">Critic review|(\d+K)</span><span class="label">Critic review')
findSC = re.compile(r'>(\d+)</span></span><span class="label"><span class="metacritic-score-label">Metascore</span>')
for i in range(8001,9301):
    print(i)
    temp_str = str(int(links[i,1]))
    while len(temp_str) != 7:
        temp_str = '0' + temp_str
    url = "https://www.imdb.com/title/tt" + temp_str + "/?ref_=ttcrv_ov"
    try:
        html = askURL(url)  # 保存获取到的网页源码
    except:
        pass
    # 2.逐一解析数据
    try:
        data = "0"
        try:
            data = re.findall(findUR,html)[0]
            if type(data) == tuple:
                if data[0] == '':
                    if data[1] == '':
                        data = data[2]
                    else:
                        data = data[1]
                else:
                    data = data[0]
        except:
            pass
        data1 = "0"
        try:
            data1 = re.findall(findCR, html)[0]
            if type(data1) == tuple:
                if data1[0] == '':
                    if data1[1] == '':
                        data1 = data1[2]
                    else:
                        data1 = data1[1]
                else:
                    data1 = data1[0]
        except:
            pass
        data2 = 0
        try:
            data2 = re.findall(findSC,html)[0]
        except:
            pass
        if data[-1] == 'K':
            data = int(float(data[:-1]) * 1000)
        else:
            data = int(data)
        if data1[-1] == 'K':
            data1 = int(float(data1[:-1]) * 1000)
        else:
            data1 = int(data1)
        data2 = int(data2)
        table[i,1] = data
        table[i,2] = data1
        table[i,3] = data2
    except:
        pd.DataFrame(table).to_csv("table1.csv")
        exit(i)
pd.DataFrame(table).to_csv("table1.csv")