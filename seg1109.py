# 分词得到分词后的文档（评论文本）
import jieba
import jieba.posseg as psg
from snownlp import SnowNLP

##以情感得分把评论文本分为两部分
input_file = 'D:/Yubo Guo/Z2进行中的研究/lda/textmining/爬虫/comments1109-573.txt'  # 输入文件路径
posi = []
nega = []
with open(input_file, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        text = line.strip()
        if text:
            s = SnowNLP(text)
            sentiment_score = s.sentiments
            if sentiment_score > 0.8:
                posi.append(i)
            else:
                nega.append(i)
print(posi)
print(len(posi))
print(nega)
print(len(nega))
83257668

##分词，形成可做LDA的documents
# 定义函数，加载txt文件
def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words


# 调用函数--原始语料
text = readtxt('D:/Yubo Guo/Z2进行中的研究/lda/textmining/爬虫/comments1109-573.txt')  # 一个文章摘要形成列表中的一个元素
print(len(text))


# 定义分词函数
def cut_word(text_use):
    # 加载用户自定义词典
    jieba.load_userdict("D:/Yubo Guo/Z2进行中的研究/lda/textmining/user_dict2.txt")
    # 加载停用词表
    stopwords = readtxt('D:/Yubo Guo/Z2进行中的研究/lda/textmining/stopwords.txt', encoding='gbk')
    checkarr = ['n']
    res = []
    for word, flag in psg.lcut(text_use):
        if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
            res.append(word)
    return " ".join(res)


# 分词_对所有评论
segged_words_total = [cut_word(x) for x in text]
print(segged_words_total)
print(len(segged_words_total))
# 抽取积极评论索引
segged_words_posi = []
for j in range(len(posi)):
    k = posi[j]
    segged_words_posi.append(segged_words_total[k])
print(segged_words_posi)
print(len(segged_words_posi))
# 抽取消极评论索引
segged_words_nega = []
for j in range(len(nega)):
    k = nega[j]
    segged_words_nega.append(segged_words_total[k])
print(segged_words_nega)
print(len(segged_words_nega))


# # 所有评论，由213删除空行得到178
f = open("sdwords_total.txt", "w", encoding='utf-8')
for line in segged_words_total:
    if line != '':
        f.write(line + '\n')
f.close()
file_path = 'sdwords_total.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file.readlines()]
print(lines)
print(len(lines))

# # 积极评论，删除空行得到135
f = open("sdwords_positive.txt", "w", encoding='utf-8')
for line in segged_words_posi:
    if line != '':
        f.write(line + '\n')
f.close()
file_path = 'sdwords_positive.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file.readlines()]
print(lines)
print(len(lines))

# # 消极评论，删除空行得到43
f = open("sdwords_negative.txt", "w", encoding='utf-8')
for line in segged_words_nega:
    if line != '':
        f.write(line + '\n')
f.close()
file_path = 'sdwords_negative.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = [line.strip() for line in file.readlines()]
print(lines)
print(len(lines))
