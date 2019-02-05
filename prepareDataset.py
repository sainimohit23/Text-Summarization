import re
import pickle
import numpy as np

train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"

def read_glove_vectors(path):
    with open(path, encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            cur_word = line[0]
            words.add(cur_word)
            word_to_vec_map[cur_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    sentence = re.sub(r"[-()\"/@;:<>{}`+=~|.!?,']", "", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]

words_to_index, index_to_words, word_to_vec_map = read_glove_vectors('glove.6B.50d.txt')
words_to_index['<UNK>'] = len(words_to_index)
words_to_index['<PAD>'] = len(words_to_index)
words_to_index['<GO>'] = len(words_to_index)
words_to_index['<EOS>'] = len(words_to_index)
index_to_words[len(index_to_words)+1] = '<UNK>'
index_to_words[len(index_to_words)+1] = '<PAD>'
index_to_words[len(index_to_words)+1] = '<GO>'
index_to_words[len(index_to_words)+1] = '<EOS>'
word_to_vec_map['<UNK>'] = np.random.uniform(low=-0.5, high=0.5, size=(50,))
word_to_vec_map['<PAD>'] = np.random.uniform(low=-0.5, high=0.5, size=(50,))
word_to_vec_map['<GO>'] = np.random.uniform(low=-0.5, high=0.5, size=(50,))
word_to_vec_map['<EOS>'] = np.random.uniform(low=-0.5, high=0.5, size=(50,))


for key,vec in word_to_vec_map.items():
    word_to_vec_map[key] = word_to_vec_map[key].astype(np.float32)



article_list = []
for article in get_text_list(train_article_path, True):
    tokens = []
    line=[]
    for word in article.strip().split():
        line.append(word)
        if word in words_to_index.keys():
            tokens.append(words_to_index[word])
        else:
            tokens.append(words_to_index['<UNK>'])
    print(line)
    article_list.append(tokens)

title_list = []
for title in get_text_list(train_title_path, True):
    tokens = []
    for word in title.strip().split():
        if word in words_to_index.keys():
            tokens.append(words_to_index[word])
        else:
            tokens.append(words_to_index['<UNK>'])
    title_list.append(tokens)

with open('processedGlove.p', 'wb') as f:
    pickle.dump((words_to_index, index_to_words, word_to_vec_map), f)
    
with open('smallData.p', 'wb') as f:
    pickle.dump((article_list, title_list), f)
    
    