import pandas as pd
import re

from gensim.models import Word2Vec

df = pd.read_csv('dataset/train_j2p.csv')
tgts = df['tgt'].tolist()
srcs = df['src'].tolist()
df = pd.read_csv('dataset/test_j2p.csv')
tgts.extend(df['tgt'].tolist())
srcs.extend(df['src'].tolist())
df = pd.read_csv('dataset/valid_j2p.csv')
tgts.extend(df['tgt'].tolist())
srcs.extend(df['src'].tolist())

def snake_case(s):
    return re.sub('([A-Z])', r'_\1', s).lstrip('_')

# func_names = []
token_names = []
for src in srcs:
    temp = []
    src = src.replace("Translate Java to Python: ", "")
    for token in src.split():
        if '_' in token:
            temp.extend(token.split('_'))
        else:
            token = snake_case(token)
            if '_' in token:
                temp.extend(token.split('_'))
            else:
                temp.append(token)
    token_names.append(temp)

for src in tgts:
    temp = []
    src = src.replace("NEW_LINE", "")
    src = src.replace("INDENT", "")
    src = src.replace("DEDENT", "")
    for token in src.split():
        if '_' in token:
            temp.extend(token.split('_'))
        else:
            token = snake_case(token)
            if '_' in token:
                temp.extend(token.split('_'))
            else:
                temp.append(token)
    token_names.append(temp)

# print(token_names)
model = Word2Vec(sentences=token_names, vector_size=300, window=5, min_count=1, workers=4)
print(model.wv.most_similar('check', topn=20))
# model.save("models/word2vec.model")

