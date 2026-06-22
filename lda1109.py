import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

cop = open(r'sdwords_total.txt', 'r', encoding='UTF-8')
train = []  
for line in cop.readlines():
    line = [word.strip() for word in line.split(' ')]
    train.append(line)  
print(train)

dictionary = Dictionary(train)
corpus = [dictionary.doc2bow(text) for text in
          train]  

lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=9, minimum_probability=0, random_state=20210607)

plot = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(plot, 'lda_topic_total_visual.html')

with open('r_topic-word.txt', 'w', encoding='UTF - 8') as file:
    for topic_id in range(lda.num_topics):
        topic_words = lda.show_topic(topic_id)
        file.write(f"Topic {topic_id + 1}:\n")
        for word, prob in topic_words:
            file.write(f"{word}: {prob}\n")
        file.write('\n')

DocumentTopicMatrix = np.zeros([492, 9])  
for id in range(len(corpus)):
    doc_top = lda.get_document_topics(corpus[id], per_word_topics=False)
    m = 0
    for x in doc_top:
        DocumentTopicMatrix[id][m] = x[1]
        m += 1
np.savetxt("r_docu-topic.txt", DocumentTopicMatrix)

def find_max_column(file_path):
    max_column_indices = []
    with open(file_path, 'r') as f:
        for line in f:
            columns = line.split()
            numbers = [float(column) for column in columns]
            max_index = numbers.index(max(numbers))
            max_column_indices.append(max_index + 1)  
    return max_column_indices
max_column_indices = find_max_column('r_docu-topic.txt')

matching_lines = []
with open('sdwords_total.txt', 'r', encoding='utf-8') as current_f:
    current_lines = current_f.readlines()
with open('sdwords_positive.txt', 'r', encoding='utf-8') as other_f:
    other_lines = other_f.readlines()
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
print(len(matching_lines))
posi_topic = []
for jk in range(len(matching_lines)):
    jkl = matching_lines[jk]
    posi_topic.append(max_column_indices[jkl])
print(len(posi_topic))
np.savetxt("r_posi_doc-topic.txt", posi_topic)


matching_lines_nega = []
with open('sdwords_total.txt', 'r', encoding='utf-8') as current_f:
    current_lines = current_f.readlines()
with open('sdwords_negative.txt', 'r', encoding='utf-8') as other_f:
    other_lines = other_f.readlines()
current_line_num = 1
other_line_num = 1
while current_line_num <= len(current_lines) and other_line_num <= len(other_lines):
    current_line = current_lines[current_line_num - 1]
    other_line = other_lines[other_line_num - 1]
    if current_line == other_line:
        matching_lines_nega.append(current_line_num - 1)
        current_line_num += 1
        other_line_num += 1
    else:
        current_line_num += 1
print(len(matching_lines_nega))
nega_topic = []
for jk in range(len(matching_lines_nega)):
    jkl = matching_lines_nega[jk]
    nega_topic.append(max_column_indices[jkl])
print(len(nega_topic))
np.savetxt("r_nega_doc-topic.txt", nega_topic)
