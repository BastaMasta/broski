import gensim.downloader as api
import numpy as np

word_vectors = api.load('glove-wiki-gigaword-100')
print(f"Vocabulary size: {len(word_vectors.index_to_key)}")

result = word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
for word, similarity in result:
    print(f"{word}: {similarity:.4f}")

print(word_vectors.most_similar('computer', topn=5))

print(word_vectors.doesnt_match("breakfast lunch dinner banana".split()))

print(word_vectors.similarity('king', 'queen'))