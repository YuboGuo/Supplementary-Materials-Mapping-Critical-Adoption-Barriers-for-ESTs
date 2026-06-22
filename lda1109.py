import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

# # 数据准备：sdwords_total.txt为已完成分词的documents集合
cop = open(r'sdwords_total.txt', 'r', encoding='UTF-8')
train = []  # 生成list of list 格式
for line in cop.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)  # list of list 格式
print(train)

# # 创建字典
dictionary = Dictionary(train)
# 根据字典和文本数据创建语料库
corpus = [dictionary.doc2bow(text) for text in
          train]  # corpus里面的存储格式（0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)

# # 训练lda模型：9个主题（经R中的计算，9为最佳；根据Griffiths2004和CaoJuan2009）
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=9, minimum_probability=0, random_state=20210607)

# # python中的可视化
plot = gensimvis.prepare(lda, corpus, dictionary)
# 保存到本地html
pyLDAvis.save_html(plot, 'lda_topic_total_visual.html')

# # 输出每个主题中关键词的概率分布：“r_topic-word.txt”文件用以保存结果
with open('r_topic-word.txt', 'w', encoding='UTF - 8') as file:
    for topic_id in range(lda.num_topics):
        topic_words = lda.show_topic(topic_id)
        file.write(f"Topic {topic_id + 1}:\n")
        for word, prob in topic_words:
            file.write(f"{word}: {prob}\n")
        file.write('\n')
print("主题-特征词概率分布保存成功")

# # 输出每个文档的文档-主题分布矩阵：“r_docu-topic.txt”文件用以保存结果
DocumentTopicMatrix = np.zeros([492, 9])  # 存储文档-主题分布,len(doc)就是文档的总个数
for id in range(len(corpus)):
    doc_top = lda.get_document_topics(corpus[id], per_word_topics=False)
    m = 0
    for x in doc_top:
        DocumentTopicMatrix[id][m] = x[1]
        m += 1
np.savetxt("r_docu-topic.txt", DocumentTopicMatrix)
print("文档-主题分布矩阵保存成功")


# # 在文档-主题矩阵中找出每个文档最大概率的主题类别
def find_max_column(file_path):
    max_column_indices = []
    with open(file_path, 'r') as f:
        for line in f:
            # 将每行拆分为列
            columns = line.split()
            # 将字符串列转换为数字列表
            numbers = [float(column) for column in columns]
            # 找到最大值的索引
            max_index = numbers.index(max(numbers))
            # 将最大值索引添加到结果列表
            max_column_indices.append(max_index + 1)  # 添加1以匹配列号（从1开始）：主题序号
    return max_column_indices
max_column_indices = find_max_column('r_docu-topic.txt')
print(max_column_indices)

# # 匹配积极评论在sdwords_total中的行号
matching_lines = []
# 读取当前文件的行号和内容
with open('sdwords_total.txt', 'r', encoding='utf-8') as current_f:
    current_lines = current_f.readlines()
# 读取另一个文件的行号和内容
with open('sdwords_positive.txt', 'r', encoding='utf-8') as other_f:
    other_lines = other_f.readlines()
# 遍历当前文件的行
current_line_num = 1
other_line_num = 1
while current_line_num <= len(current_lines) and other_line_num <= len(other_lines):
    current_line = current_lines[current_line_num - 1]
    other_line = other_lines[other_line_num - 1]
    if current_line == other_line:
        matching_lines.append(current_line_num - 1)
        current_line_num += 1
        other_line_num += 1
    else:
        current_line_num += 1
# 输出匹配行的行号
print('积极评论匹配的序号', matching_lines)
print(len(matching_lines))
# 求得积极评论中每个文档的最大可能主题（逐个）的主题序号
posi_topic = []
for jk in range(len(matching_lines)):
    jkl = matching_lines[jk]
    posi_topic.append(max_column_indices[jkl])
print('积极评论每个文档对应的最大可能主题的序号', posi_topic)
print(len(posi_topic))
np.savetxt("r_posi_doc-topic.txt", posi_topic)


# # 匹配消极评论在sdwords_total中的行号
matching_lines_nega = []
# 读取当前文件的行号和内容
with open('sdwords_total.txt', 'r', encoding='utf-8') as current_f:
    current_lines = current_f.readlines()
# 读取另一个文件的行号和内容
with open('sdwords_negative.txt', 'r', encoding='utf-8') as other_f:
    other_lines = other_f.readlines()
# 遍历当前文件的行
current_line_num = 1
other_line_num = 1
while current_line_num <= len(current_lines) and other_line_num <= len(other_lines):
    current_line = current_lines[current_line_num - 1]
    other_line = other_lines[other_line_num - 1]
    # 比较当前行和另一个文件的行
    if current_line == other_line:
        matching_lines_nega.append(current_line_num - 1)
        current_line_num += 1
        other_line_num += 1
    else:
        current_line_num += 1
# 输出匹配行的行号
print('消极评论匹配', matching_lines_nega)
print(len(matching_lines_nega))
# 求得消极评论中每个文档的最大可能主题（逐个）的主题序号
nega_topic = []
for jk in range(len(matching_lines_nega)):
    jkl = matching_lines_nega[jk]
    nega_topic.append(max_column_indices[jkl])
print('消极评论每个文档对应的最大可能主题的序号', nega_topic)
print(len(nega_topic))
np.savetxt("r_nega_doc-topic.txt", nega_topic)
