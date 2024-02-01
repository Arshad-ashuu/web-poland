from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from collections import Counter

app = Flask(__name__)

dt = pd.read_json("small_ds.json")
X, Y = dt.code, dt.language
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
pattern = r"""\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"']"""
vectorizer = TfidfVectorizer(token_pattern=pattern)
x_train_tf = vectorizer.fit_transform(x_train)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1500,batch_size=64)  
mnb = MultinomialNB(alpha=0.1)  
rm = RandomForestClassifier(max_depth = None, random_state = 42, n_estimators=100) 

param_grid_mnb = {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}
grid_search_mnb = GridSearchCV(mnb, param_grid_mnb, cv=5, scoring='accuracy')

grid_search_mnb.fit(x_train_tf, y_train)
best_params_mnb = grid_search_mnb.best_params_
best_mnb_model = grid_search_mnb.best_estimator_

rm.fit(x_train_tf, y_train)
mlp.fit(x_train_tf, y_train)
best_mnb_model.fit(x_train_tf, y_train)


def Testing(test_code):
    avg = []
    detected_languages = []
    test_code = vectorizer.transform([test_code])
    pred_lang_rf = rm.predict(test_code)[0]
    avg.append(pred_lang_rf)
    pred_lang_mnb = best_mnb_model.predict(test_code)[0]
    avg.append(pred_lang_mnb)
    pred_lang_mlp = mlp.predict(test_code)[0]
    avg.append(pred_lang_mlp)
    answer = max(Counter(avg), key=Counter(avg).get)
    for detect in avg:
        if detect != answer:
            x = detect
            detected_languages.append(x)
        else:
            pass
    return answer, detected_languages

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
            result, detected_languages = Testing(file_content)
            return render_template('result.html', n=file_content, p_rf=result, av=detected_languages)
        else:
            result, detected_languages = Testing(code)
            return render_template('result.html', n=code, p_rf=result, av=detected_languages)
    else:
        return render_template('result.html')

#uncomment below lines for local development
#if __name__ == "__main__":
#     app.run(debug=True)
