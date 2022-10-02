#!/usr/local/bin/python
#-*- coding:utf-8 -*-
import itchat
from itchat.content import *
import requests
import json

@itchat.msg_register(TEXT)
def reply_text(msg):
    from_text = msg['Text']
    # 消息带有 ‘#’ 前缀为翻译
    if from_text[0] == '#':
        to_text = baidu_trans(from_text[1:])
        itchat.send(to_text, msg['FromUserName'])
    else:
        to_text = baidu_trans(from_text)
        itchat.send(to_text,msg['FromUserName'])

url2 = 'http://fanyi.baidu.com/v2transapi'
def baidu_trans(info):
    keywords = {
        'from': 'zh',
        'to': 'en',
        'query': info,

    }
    req = requests.post(url2, keywords)
    data = req.json()
    try:
        result = data['dict_result']['simple_means']['word_means']
        return ';'.join(result)

    except:

        return data['trans_result']['data'][0]['dst']


if __name__ == '__main__':
    itchat.auto_login(hotReload=True)
    itchat.run()