import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import gensim.downloader as api
import numpy as np

word_vectors = api.load('glove-wiki-gigaword-100')
print(f"Vocabulary size: {len(word_vectors.index_to_key)}")

sports_words = ['football', 'soccer', 'tennis', 'basketball', 'cricket', 'goal', 'player', 'team', 'coach',
'score']
sports_vectors = np.array([word_vectors[word] for word in sports_words])

pca = PCA(n_components=2)
sports_2d = pca.fit_transform(sports_vectors)
plt.figure(figsize=(8,6))
for i, word in enumerate(sports_words):
    plt.scatter(sports_2d[i,0], sports_2d[i,1])
    plt.annotate(word, (sports_2d[i,0], sports_2d[i,1]))
plt.title("PCA Visualization of Sports Words")
plt.show()


tsne = TSNE(n_components=2, random_state=42, perplexity=5)
sports_tsne = tsne.fit_transform(sports_vectors)
plt.figure(figsize=(8,6))
for i, word in enumerate(sports_words):
    plt.scatter(sports_tsne[i,0], sports_tsne[i,1])
    plt.annotate(word, (sports_tsne[i,0], sports_tsne[i,1]))
plt.title("t-SNE Visualization of Sports Words")
plt.show()

def get_similar_words(word):
    try:
        result = word_vectors.most_similar(word, topn=5)
        for w, sim in result:
            print(f"{w}: {sim:.4f}")
    except KeyError:
        print(f"'{word}' not in vocabulary!")

get_similar_words('football')