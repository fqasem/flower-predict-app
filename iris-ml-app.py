import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type based on input parameters such as sepal lenght and width and petal length and width from the sidebar!
""")

st.sidebar.header('User Input Parameters')
st.sidebar.write("""
    Sepal length and width and petal length and width are the features used to predict the type of Iris flower.
    The dataset contains three classes of iris plants: Iris Setosa, Iris Versicolor, and Iris Virginica.
    The features are:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width
    The target variable is the type of iris flower, which can be one of the following:
    - Iris Setosa
    - Iris Versicolor
    - Iris Virginica
    The app uses the Iris dataset from the `sklearn.datasets` module, which is a well-known dataset for classification tasks.
    The dataset contains 150 samples, with 50 samples for each of the three classes.
    The model used for prediction is a Random Forest Classifier, which is a type of ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.
    The app uses the `RandomForestClassifier` from the `sklearn.ensemble` module to train the model on the iris dataset and make predictions based on user input.            
T""")

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
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
