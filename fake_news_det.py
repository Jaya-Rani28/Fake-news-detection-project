from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

tfidf_vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
loaded_model = pickle.load(open('final_model.pkl', 'rb'))


app = Flask(__name__)   

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        news = str(request.form['news'])
        print(news)
        pred = loaded_model.predict(tfidf_vectorizer.transform([news]))[0]
        print(pred)
        return render_template('prediction.html', prediction_text="News headline is : {}".format(pred))
    else:
        return render_template("prediction.html", prediction_text="Prediction of the news")

if __name__ == '__main__':
    app.run(debug=True)