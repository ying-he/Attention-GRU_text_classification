# -*- coding: utf-8 -*-
# Author: HeYing
# Creation Date: 2019-05-21

# import csv
import pickle
import gensim
import nltk
import pickle
import numpy as np
import time
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import defaultdict
stopwordDict = set(stopwords.words("english"))


def save_variable(v, filename):
    f = open(filename, "wb")
    pickle.dump(v, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, "rb")
    r = pickle.load(f)
    f.close()
    return r


traindata = []
testdata = []
# __label__1 corresponds to 1- and 2-star reviews
# __label__2 corresponds to 4- and 5-star reviews
label1 = "__label__1"
label2 = "__label__2"
Nlabel = len(label1)
with open("/Users/heying/Downloads/amazonreviews/train.ft.txt", "r") as ftrain:
    cnt = 0
    for line in ftrain:
        if cnt % 10 == 0:
            content = line.replace("\n", "").strip()[Nlabel:]
            if line[:Nlabel] == label1:
                label = 0
            elif line[:Nlabel] == label2:
                label = 1
            else:
                label = "null"
            traindata.append("\t".join([content, str(label)]))
        # 2 columns: text content, sentimental label
        cnt += 1
print("Samples of training set is %d." % cnt)

testcnt = 0
with open("/Users/heying/Downloads/amazonreviews/test.ft.txt", "r") as ftest:
    for line in ftest:
        if testcnt % 10 == 0:
            content = line.replace("\n", "").strip()[Nlabel:]
            if line[:Nlabel] == label1:
                label = 0
            elif line[:Nlabel] == label2:
                label = 1
            else:
                label = "null"
            testdata.append("\t".join([content, str(label)]))
        # 4 columns: id, sentimental label, text content, is training or not.
        testcnt += 1
print("Samples of testing set is %d." % testcnt)

# 将原始文本文件的标签处理后存储到新的文本文件中, 36w训练集数据，4w测试集数据
with open("/Users/heying/Documents/EDA_Tue_Liumiao/coursePaper/rnn_text_classification-master/data/amazonTrain.txt", "w") as T1:
    T1.write("\n".join(traindata))
    # writer = csv.writer(T1)
    # writer.writerow(["id", "label", "content", "type"])
    # writer.writerows(traindata)
with open("/Users/heying/Documents/EDA_Tue_Liumiao/coursePaper/rnn_text_classification-master/data/amazonTest.txt", "w") as T2:
    T2.write("\n".join(testdata))
    # writer = csv.writer(T2)
    # writer.writerow(["id", "label", "content", "type"])
    # writer.writerows(testdata)

if __name__ == "__main__":
    # 加载训练好的word2vec模型
    w2v_model = gensim.models.Word2Vec.load("/Users/heying/Downloads/w2vData/full_word2vec.model")
    # 进行文本预处理，先计算训练集和测试集文件中，每一行包含多少句话，每句话包含多少个单词
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')  # 分句器
    word_tokenizer = RegexpTokenizer(r'\w{2,}')  # 分词器
    num_sents_words = []  # 元素[num_sentences, [num_words in each sentence], max_word in one sentence, mean words in sentences]
    all_words_embeddings = {}  # 用于建立总词典
    all_words = []
    word_freq = defaultdict(int)  # 记录每个单词及其出现的频率
    words_notin_model = {}
    docs = []  # 用于保存所有的文档矩阵，每个文档为一个矩阵，一行为一句话，行的元素为单词，每个文档不超过20句话，每句话不超过40个词，多余的截去
    traindocs = []
    trainlabels = []
    testdocs = []
    testlabels = []
    max_sents = 20  # 每篇评论包含20个句子
    max_words = 40  # 每个句子包含40个单词

    process_start = time.asctime(time.localtime(time.time()))
    with open("/Users/heying/Documents/EDA_Tue_Liumiao/coursePaper/rnn_text_classification-master/data/amazonTrain.txt", "r") as f1:
        for line in f1:
            review, label = line.replace("\n", "").split("\t")
            label = int(label)
            label_list = [0] * 2
            label_list[label] = 1
            trainlabels.append(label_list)  # 存储训练集标签信息
            review = review.lower()
            # 对评论文本中的一些缩略词作处理
            review = review.replace("\'t", "t")
            review = review.replace("\'s", "s")
            review = review.replace("\'re", "re")
            review = review.replace("\'ll", " will")
            review = review.replace("\'d", " would")
            sents = sent_tokenizer.tokenize(review)
            words_in_sents = [word_tokenizer.tokenize(each) for each in sents]
            one_wordlist = sum(words_in_sents, [])  # 将嵌套的单词列表展开为单个列表
            all_words += one_wordlist
            num_words = [len(each_wordlist) for each_wordlist in words_in_sents]
            num_sents_words.append([len(num_words), num_words, max(num_words), np.mean(num_words)])

            # 整理单词及其对应的词向量
            for each_word in one_wordlist:
                if each_word not in word_freq:
                    word_freq[each_word] = 1
                    if each_word not in w2v_model:
                        if each_word not in words_notin_model:
                            words_notin_model[each_word] = 1
                        else:
                            words_notin_model[each_word] += 1
                    else:
                        all_words_embeddings[each_word] = w2v_model.wv[each_word]
                else:
                    word_freq[each_word] += 1
                    if each_word not in w2v_model:
                        words_notin_model[each_word] += 1

            # 将每篇评论整理为 20 * 40 的矩阵
            one_doc = []
            if len(words_in_sents) < max_sents:
                for each_sent in words_in_sents:
                    if len(each_sent) < max_words:
                        one_doc.append(each_sent + ["UNK"]*(max_words-len(each_sent)))
                    else:
                        one_doc.append(each_sent[:max_words])
                one_doc += [["UNK"]*max_words]*(max_sents-len(words_in_sents))
            else:
                for k in range(max_sents):
                    if len(words_in_sents[k]) < max_words:
                        one_doc.append(words_in_sents[k] + ["UNK"]*(max_words-len(words_in_sents[k])))
                    else:
                        one_doc.append(words_in_sents[k][:max_words])
            traindocs.append(one_doc)
            # docs.append(np.array(one_doc, dtype=object))
    # 保存处理好的训练集文本内容
    save_variable((traindocs, trainlabels), "/Users/heying/Downloads/w2vData/v_trainset")

    with open("/Users/heying/Documents/EDA_Tue_Liumiao/coursePaper/rnn_text_classification-master/data/amazonTest.txt", "r") as f2:
        for line in f2:
            review, label = line.replace("\n", "").split("\t")
            label = int(label)
            label_list = [0] * 2
            label_list[label] = 1
            testlabels.append(label_list)  # 存储测试集标签信息
            review = review.lower()
            # 对评论文本中的一些缩略词作处理
            review = review.replace("\'t", "t")
            review = review.replace("\'s", "s")
            review = review.replace("\'re", "re")
            review = review.replace("\'ll", " will")
            review = review.replace("\'d", " would")
            sents = sent_tokenizer.tokenize(review)
            words_in_sents = [word_tokenizer.tokenize(each) for each in sents]
            one_wordlist = sum(words_in_sents, [])  # 将嵌套的单词列表展开为单个列表
            all_words += one_wordlist
            num_words = [len(each_wordlist) for each_wordlist in words_in_sents]
            num_sents_words.append([len(num_words), "_".join(list(map(str,num_words))), max(num_words), np.mean(num_words)])
            for each_word in one_wordlist:
                if each_word not in word_freq:
                    word_freq[each_word] = 1
                    if each_word not in w2v_model:
                        if each_word not in words_notin_model:
                            words_notin_model[each_word] = 1
                        else:
                            words_notin_model[each_word] += 1
                    else:
                        all_words_embeddings[each_word] = w2v_model.wv[each_word]
                else:
                    word_freq[each_word] += 1
                    if each_word not in w2v_model:
                        words_notin_model[each_word] += 1

            # 将每篇评论整理为 20 * 40 的矩阵
            one_doc = []
            if len(words_in_sents) < max_sents:
                for each_sent in words_in_sents:
                    if len(each_sent) < max_words:
                        one_doc.append(each_sent + ["UNK"] * (max_words - len(each_sent)))
                    else:
                        one_doc.append(each_sent[:max_words])
                one_doc += [["UNK"] * max_words] * (max_sents - len(words_in_sents))
            else:
                for k in range(max_sents):
                    if len(words_in_sents[k]) < max_words:
                        one_doc.append(words_in_sents[k] + ["UNK"] * (max_words - len(words_in_sents[k])))
                    else:
                        one_doc.append(words_in_sents[k][:max_words])
            testdocs.append(one_doc)
            # docs.append(np.array(one_doc, dtype=object))
    # 保存处理好的测试集文本内容
    save_variable((testdocs, testlabels), "/Users/heying/Downloads/w2vData/v_testset")

    print("Size of the training set is: %d, %d" % (len(traindocs), len(trainlabels)))
    print("Size of the testing set is: %d, %d" % (len(testdocs), len(testlabels)))
    print("The vocabulary size is %d" % len(word_freq))
    print("Number of all words in all the docs is %d" % len(all_words))
    unique_words = list(set(all_words))
    print("Number of unique words in all the docs is %d" % len(unique_words))
    total_word_freq = 0
    for word in word_freq:
        total_word_freq += word_freq[word]
    for key in words_notin_model:
        all_words_embeddings[key] = np.random.randn(400)
        # 即对于未登陆词，随机生成一个正态分布的400维数组
    all_words_embeddings["UNK"] = np.random.rand(400)
    # 对于长度不够的句子，使用一个400维的随机数组（0-1之间）来填补
    # # 存储词频文件
    # with open("/Users/heying/Downloads/w2vData/wordSentFreq.txt", "w") as wf:
    #     items = list(map(str, num_sents_words))
    #     wf.writelines("\n".join(items))
    print("The word embedding size is %d" % len(all_words_embeddings))
    # 保存数据集中所有可能出现的词嵌入向量字典
    save_variable(all_words_embeddings, "/Users/heying/Downloads/w2vData/word_embedding")
    process_finish = time.asctime(time.localtime(time.time()))

    # embedding_start = time.asctime(time.localtime(time.time()))
    # # 存储训练集文档对应的词向量信息
    # print("Start processing embeddings...")
    # print("Time now is: ", time.asctime(time.localtime(time.time())))
    # traindocs_embeddings = [[[0.0 for i in range(40)]for j in range(20)]for k in range(360000)]
    # # 存储训练集文档对应的词向量信息
    # testdocs_embeddings = [[[0.0 for i in range(40)]for j in range(20)]for k in range(40000)]
    # for i in range(len(traindocs)):
    #     if i % 1000 == 0:
    #         print("Processing training docs NO.%d" % i)
    #         print(time.asctime(time.localtime(time.time())))
    #     for j in range(len(traindocs[i])):
    #         for k in range(len(traindocs[i][j])):
    #             traindocs_embeddings[i][j][k] = list(all_words_embeddings[traindocs[i][j][k]])
    # for i in range(len(testdocs)):
    #     if i % 1000 == 0:
    #         print("Processing testing docs NO.%d" % i)
    #         print(time.asctime(time.localtime(time.time())))
    #     for j in range(len(testdocs[i])):
    #         for k in range(len(testdocs[i][j])):
    #             testdocs_embeddings[i][j][k] = list(all_words_embeddings[testdocs[i][j][k]])
    # print("Start saving embedding docs ... Time now is:  ")
    # print(time.asctime(time.localtime(time.time())))
    # save_variable((traindocs_embeddings, trainlabels), "/Users/heying/Downloads/w2vData/v_trainset_embeddings")
    # save_variable((testdocs_embeddings, testlabels), "/Users/heying/Downloads/w2vData/v_testset_embeddings")
    # print("Saving is finished.")
    # embedding_finish = time.asctime(time.localtime(time.time()))
    # 输出程序运行的时间细节
    print("\nTime spending details: ")
    print("Start to process docs at time point: ")
    print(process_start, ".")
    print("Finish processing at time point: ")
    print(process_finish, ".")
    # print("Start to save embedding docs at time point: ")
    # print(embedding_start, ".")
    # print("Finish saving at time point: ")
    # print(embedding_finish, ".")


