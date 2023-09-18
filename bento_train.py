import bentoml

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
iris = load_iris()
X = iris.data[:, :4]
Y = iris.target
model.fit(X, Y)

bento_model = bentoml.sklearn.save_model('kneighbors', model)
print(f"Model saved: {bento_model}")