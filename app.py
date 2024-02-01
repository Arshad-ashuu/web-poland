from flask import Flask, render_template, request,flash,redirect, url_for
import pandas as pd
from werkzeug.utils import secure_filename
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier


app = Flask(__name__)
#Data Preprocessing
dt = pd.read_json('small_ds.json')
X, Y = dt.code, dt.language
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
pattern = r"""\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"']"""
vectorizer = TfidfVectorizer(token_pattern=pattern)
x_train_tf = vectorizer.fit_transform(x_train)

model = RandomForestClassifier(max_depth=10, random_state=0)
mnb = MultinomialNB()
mlp = MLPClassifier(max_iter=1600)
model.fit(x_train_tf, y_train)
mnb.fit(x_train_tf, y_train)
mlp.fit(x_train_tf, y_train)


# def read_file(open_file):
#     with open(open_file, 'r', encoding='utf-8') as file:
#         read_content = file.read()
#     return read_content
#predicition
def Testing(test_code):
    avg = []
    test_code = vectorizer.transform([test_code])
    pred_lang = (model.predict(test_code)[0])
    avg.append(pred_lang)
    pred_lang = (mnb.predict(test_code)[0])
    avg.append(pred_lang)
    pred_lang = (mlp.predict(test_code)[0])
    avg.append(pred_lang)
    answer = max(Counter(avg), key=Counter(avg).get)
    return answer   

#frontend
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def res():
    if request.method == 'POST':
        code = request.form['Name']
        file = request.files['upload_file']

        if file:
            file_content = file.read()
            result = Testing(file_content)
            return render_template('result.html', n=file_content, p_rf=result)
        else:
            result = Testing(code)
            return render_template('result.html', n=code, p_rf=result)
    else:

        return render_template('result.html')



   
   # for loacl development uncomment these 2 lines  
#if __name__ == "__main__":
#    app.run(debug=True)
