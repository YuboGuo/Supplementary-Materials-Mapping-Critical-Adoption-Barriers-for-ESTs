import jieba
import jieba.posseg as psg
from snownlp import SnowNLP

input_file = 'D:/Yubo Guo/Z2/lda/textmining/crawler/comments1109-573.txt'  
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


def readtxt(filepath, encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r', encoding=encoding).readlines()]
    return words

text = readtxt('D:/Yubo Guo/Z2/lda/textmining/crawler/comments1109-573.txt')  
print(len(text))


def cut_word(text_use):
    jieba.load_userdict("D:/Yubo Guo/Z2/lda/textmining/user_dict2.txt")
    stopwords = readtxt('D:/Yubo Guo/Z2/lda/textmining/stopwords.txt', encoding='gbk')
    checkarr = ['n']
    res = []
    for word, flag in psg.lcut(text_use):
        if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
            res.append(word)
    return " ".join(res)


segged_words_total = [cut_word(x) for x in text]
print(segged_words_total)
print(len(segged_words_total))
segged_words_posi = []
for j in range(len(posi)):
    k = posi[j]
    segged_words_posi.append(segged_words_total[k])
print(segged_words_posi)
print(len(segged_words_posi))
segged_words_nega = []
for j in range(len(nega)):
    k = nega[j]
    segged_words_nega.append(segged_words_total[k])
print(segged_words_nega)
print(len(segged_words_nega))

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
