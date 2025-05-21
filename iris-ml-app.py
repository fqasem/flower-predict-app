import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type based on input parameters such as sepal lenght and width and petal length and width from the sidebar!
""")

st.sidebar.header('User Input Parameters')
st.sidebar.markdown("""
**Features used to predict the type of Iris flower:**
- Sepal length
- Sepal width
- Petal length
- Petal width

**Classes of iris plants:**
- Iris Setosa
- Iris Versicolor
- Iris Virginica

The app uses the Iris dataset from `sklearn.datasets`, a well-known dataset for classification tasks (150 samples, 50 per class).  
Features are continuous values; the target is categorical.

The model: `RandomForestClassifier` from `sklearn.ensemble` is used to train and make predictions based on user input.
""")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4, help="The length of the sepal in centimeters.")
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4, help="The width of the sepal in centimeters.")
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3, help="The length of the petal in centimeters.")
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2, help="The width of the petal in centimeters.")
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels')
st.write(iris.target_names)

st.subheader('Prediction')
st.write('The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates. ' \
'That is, the predicted class is the one with highest mean probability estimate across the trees.')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write('The predicted class probabilities will tell you what is the probability of being in one of the three clases. ' \
         'for example, if the predicted class is SETOSA, what is the probability of it being SETOSA?')
st.write('A 100 means the probability is 100%')

st.write(prediction_proba)
