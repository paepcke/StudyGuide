#This uses K-means on the files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import enchant

documents = []

#opens the sql file
with open('../../input/new.sql', 'r') as f:
    documents = [line.strip() for line in f]
vectorizer = TfidfVectorizer(stop_words='english')
documents = documents[:300]
X = vectorizer.fit_transform(documents)

true_k = 100
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :20]:
        #if d.check(terms[ind]):
        print(' %s' % terms[ind]),
    print


print("\n")
print("Prediction")

#tests for K-means
Y = vectorizer.transform(["south america"])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["progrmaming"])
prediction = model.predict(Y)
print(prediction)
