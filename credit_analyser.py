from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from csv import reader
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from flask_cors import CORS, cross_origin

data = pd.read_csv('german_credit_data.csv')
print(data.head(5))

t_features = []
for amt, age, time, h in zip(list(data['Credit amount']), list(data['Age']), list(data['Duration']), list(data['Housing'])):
    t_features.append([amt, age, time, h])

t_labels = list(data['Job'])
train_features = t_features[:900]
test_features = t_features[900:]
print('test feature--- ', t_features[900:])
train_labels = t_labels[:900]
test_labels = t_labels[900:]

clf = RandomForestClassifier(n_estimators=200, max_depth=100)
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)

print (pred)
print ('Accuracy:', {accuracy_score(test_labels,pred)})

pickle.dump(clf, open('clf.pkl','wb'))

from flask import Flask, request, jsonify
app = Flask(__name__)
CORS(app, origins="http://localhost:3000")

@app.route('/getPrediction', methods=['POST'])
@cross_origin(origins='http://localhost:3000',methods=['POST'],headers=['Content-Type'])
def chat():
    print('---->', list(request.json.values()))
    output = predictor(list(list(request.json.values())))
    print('-output->' + str(output))
    response = jsonify({"claim_probability": int(output)})

    print('response-->', response.headers)
    return response

def predictor(prediction_input):
    prediction_input = np.array(prediction_input).reshape(1, -1)  # Reshape the input
    pred = clf.predict(prediction_input)
    return pred[0]

if __name__ == '__main__':
    app.run(debug=True)
