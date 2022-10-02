#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import pathlib

import pandas as pd
import numpy as np

def get_friends(save=False):
    import itchat
    itchat.auto_login(hotReload=True)
    friends = itchat.get_friends(update=True)[0:]   # 核心：得到friends列表集，内含很多信息
    data = pd.DataFrame(friends)
    if save:
        data.to_excel('friends.xls')
    return data

def gender():
    if pathlib.Path('friends.xls').exists():
        data = pd.read_excel('friends.xls')
    else:
        data = get_friends()
    xs = data['Sex']
    key = '性别'

    c = collections.Counter(xs)
    ns = c[1], c[2], c[0]
    labels = '男', '女', '未知'

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['simhei'] 
    matplotlib.rcParams['font.family']='sans-serif'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pie(ns, labels=labels, autopct='%3.1f%%', shadow=True, explode=(0.1,)*2 + (0.2,))
    ax.set_title(f'微信好友{key}分布(总数：{len(xs)})')
    plt.show()


def province():
    if pathlib.Path('friends.xls').exists():
        data = pd.read_excel('friends.xls')
    else:
        data = get_friends()

    key = '省份'
    xs = data['Province']

    c = collections.Counter(xs)
    K = 9
    c = c.most_common(K-1)
    ns = [n for ci, n in c]
    labels = [ci if isinstance(ci, str) else '未知' for ci, n in c]
    nn = sum([n for k, n in c if k not in labels])
    ns.append(nn)
    labels.append('其他')

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['simhei'] 
    matplotlib.rcParams['font.family']='sans-serif'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.pie(ns, labels=labels, autopct='%3.1f%%', shadow=True, explode=(0.1,)*5 + (0.2,)*4)
    ax.set_title(f'微信好友{key}分布(总数：{len(xs)})')
    plt.show()

def signature2wc(mask_image='longmao.jpeg'):
    import re, wc
    if pathlib.Path('friends.xls').exists():
        data = pd.read_excel('friends.xls')
    else:
        data = get_friends()
    tList=[]
    for signature in data["Signature"]:
        if isinstance(signature, str):
            signature = signature.replace(" ", "").replace("span", "").replace("class", "").replace("emoji", "")
            rep = re.compile(r"1f\d.+")
            signature = rep.sub("", signature)
            tList.append(signature)
    text = " ".join(tList)

    wordcloud = wc.text2wc(text, mask_image=mask_image)
    wordcloud.to_file('wc_sig.png')

def topic():
    import jieba
    import gensim

    if pathlib.Path('friends.xls').exists():
        data = pd.read_excel('friends.xls')
    else:
        data = get_friends()

    STOPWORDS = {'“', '”', '，', '。', '：', '；', '！', '？', '）', '（', '、', 'á', 'ã', 'o', 'v', ':', ' ', '-', '^', '』','『', '\n',
    '的', '在', '要', '和', '做', '了', "也", '就', '都','.', ';', '"', ',', '…', '<', '>', '~', '/', '=','―', '～', '\'', '—', 'span', 'emoji', '不'}
    texts = []
    corpus = []
    for s in data['Signature']:
        if isinstance(s, str) and s[0]!='<':
            texts.append([w for w in jieba.cut(s) if w not in STOPWORDS])

    id2word = gensim.corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, update_every=1, passes=1)

    wordm = []
    weightm = []
    for k, topic in enumerate(lda.get_topics().tolist()):
        inds = np.argsort(topic)
        words = []
        weights= []
        for i in inds[-1:-15:-1]:
            words.append(id2word[i])
            weights.append(topic[i])
        words.append('其他')
        weights.append(1-np.sum(weights))
        wordm.append(words)
        weightm.append(weights)

    for k, ws in enumerate(wordm, 1):
        print(f"主题{k}: {'-'.join(ws)}")


if __name__ == '__main__':
    # get_friends(save=True)
    signature2wc(mask_image='222.jpg')
    # topic()
    # 