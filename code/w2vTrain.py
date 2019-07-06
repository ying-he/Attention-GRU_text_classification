# -*- coding: utf-8 -*-
# Author: HeYing
# Creation Date: 2019-05-29

from __future__ import print_function
import six
import gensim
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
# from gensim.models.word2vec import PathLineSentences
from gensim.models.word2vec import LineSentence
import time

if __name__ == '__main__':

    # 以下是wiki文本内容的提取过程
    # inp = "/Users/heying/Downloads/w2vData/enwiki-20190520-pages-articles-multistream1.xml-p10p30302.bz2"
    # outp = "/Users/heying/Downloads/w2vData/out_wiki.en.txt"
    # inp = "./w2vData/enwiki-20190520-pages-articles-multistream.xml.bz2"
    inp = "/Users/heying/Downloads/w2vData/enwiki-20190520-pages-articles-multistream.xml.bz2"
    # outp = "./w2vData/out_wiki.en.txt"
    outp = "/Users/heying/Downloads/w2vData/out_wiki.en.txt"

    print("Starting extracting corpus, time now is: ")
    print(time.asctime(time.localtime(time.time())))
    space = " "
    i = 0
    output = open(outp, 'w', encoding="utf-8")
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if(i % 100000 == 0):
            print(i)
        if six.PY3:
            output.write(' '.join(text) + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
    output.close()
    print("The Extraction of the corpus is completed. Time: ")
    print(time.asctime(time.localtime(time.time())))

    # 以下是word2vec训练过程，基于已经提取好的文本内容
    # 指定训练模型的输入和输出路径
    input1 = "/Users/heying/Downloads/w2vData/out_wiki.en.txt"
    output1 = "/Users/heying/Downloads/w2vData/word2vec.model"
    output2 = "/Users/heying/Downloads/w2vData/vector.model"
    # input1 = "./w2vData/out_wiki.en.txt"
    # output1 = "./w2vData/word2vec.model"
    # output2 = "./w2vData/vector.model"
    print("Starting training a word2vec model, time now is: ")
    print(time.asctime(time.localtime(time.time())))
    # sentences = []
    # with open(input1, 'r', encoding='utf8', errors='ignore') as f:
    #     for line in f:
    #         if " " in line:
    #             sentences.append(line.replace("\n", "").split(" "))
    # model = Word2Vec(size=200, window=5, min_count=5, workers=4)  # 定义word2vec 对象
    # model.build_vocab(sentences)  # 建立初始训练集的词典
    # model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)  # 模型训练
    model = Word2Vec(LineSentence(input1), size=400, window=10, min_count=5, workers=4)
    print("Training completed, now trying to save the trained word2vec model ...")
    model.save(output1)  # 模型保存
    model.wv.save_word2vec_format(output2, binary=False)  # 词向量保存
    print("Saving is completed. Time: ")
    print(time.asctime(time.localtime(time.time())))

    # # 以下是已训练模型的加载和使用
    # model = gensim.models.Word2Vec.load("/Users/heying/Downloads/w2vData/word2vec.model")
    # helloVec = model["apple"]


