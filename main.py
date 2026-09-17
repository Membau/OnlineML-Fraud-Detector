from river import datasets
dataset = datasets.Bananas()

from river import linear_model
model = linear_model.LogisticRegression()

for x,y in dataset:
    print (x,y)
    print(model.predict_proba_one(x))
    model.learn_one(x, y)
    print(model.predict_proba_one(x))