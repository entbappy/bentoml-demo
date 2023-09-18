import bentoml

clf = bentoml.sklearn.get('kneighbors:latest').to_runner()
clf.init_local()
print(clf.predict.run([[2,3,4,5]]))