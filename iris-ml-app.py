import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import io
import matplotlib.pyplot as plt
import seaborn as sns


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
#st.write(iris.target_names)

# Define colors for each class
colors = ['green', 'red', 'blue']


# Build a table using Markdown for class labels
table_md = "| Class Label |\n|:-----------:|\n"

for name, color in zip(iris.target_names, colors):
    table_md += f"| <span style='color:{color}'>{name.capitalize()}</span>|\n"

st.markdown(table_md, unsafe_allow_html=True)


# Create a DataFrame with target_names and corresponding colors

#labels_df = pd.DataFrame({
 #   'Class Label': iris.target_names,
  #  'Color': colors
#})


#for _, row in labels_df.iterrows():
 #   table_md += f"| <span style='color:{row['Color']}'>{row['Class Label'].capitalize()}</span>|\n"






st.subheader('Prediction')
st.write('The predicted class of an input sample is a vote by the trees in the forest, weighted by their probability estimates. ' \
'That is, the predicted class is the one with highest mean probability estimate across the trees.')
#st.write(iris.target_names[prediction])

# Build a table using Markdown for the predicted value
predicted_index = prediction[0]
predicted_label = iris.target_names[predicted_index].capitalize()
predicted_color = colors[predicted_index]

prediction_table_md = "| Prediction |\n|:-----------:|\n"
prediction_table_md += f"| <span style='color:{predicted_color}'>{predicted_label}</span> |\n"

st.markdown(prediction_table_md, unsafe_allow_html=True)





st.subheader('Prediction Probability')
st.info("""
**Prediction Probability Explanation**

- The prediction probability shows how confident the model is that your input belongs to each Iris class.
- For each class (Setosa, Versicolor, Virginica), the model gives a probability between 0 and 1.
- The highest probability indicates the predicted class.
- For example, if the probabilities are `[0.01, 0.05, 0.94]`, the model is 94% confident the flower is Virginica.
- All probabilities add up to 1 (or 100%).
""")

st.write(prediction_proba)




#st.markdown("""
#### Exploring the iris data set 
#<span title="This section explores the iris dataset, showing features, statistics, and grouped summaries.">ℹ️</span>
#""", unsafe_allow_html=True)

st.subheader("Exploring the iris data set")
st.write("This section explores the iris dataset, showing features, statistics, and grouped summaries.")

iris_df = pd.DataFrame(data = iris['data'], columns = iris['feature_names'])
iris_df['Iris type'] = iris['target']
iris_df['Iris name'] = iris_df['Iris type'].apply(lambda x: 'Setosa' if x == 0 else ('Versicolor' if x == 1 else 'Virginica'))

st.markdown("<span style='color:green'>**Loaded the features from the iris dataset:**</span>", unsafe_allow_html=True)
st.write(iris_df.head())

st.markdown("<span style='color:green'>**Let's see some statistics:**</span>", unsafe_allow_html=True)
#st.write(iris_df.info())

# Capture the output of df.info()
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

# Display the output in Streamlit
st.text(s)


st.markdown("<span style='color:green'>**Dataset described:**</span>", unsafe_allow_html=True)
st.write(iris_df.describe())


st.markdown("<span style='color:green'>**Let's group by iris names:**</span>", unsafe_allow_html=True)
st.write(iris_df.groupby(['Iris name']).describe())

st.info(
    "We can observe that there appears to be a distinction between the sizes of each type of iris. "
    "The mean values of sepal length of each flower type are 5, 5.9, and 6.5 (cm), resulting in an easy distinction, "
    "while all the other measurements follow similar trends. Although some min values might be close, some max values show differences. "
    "For example, the width of a versicolor petal is 3 times the width of the setosa one and 0.7 mm less than the virginica one."
)
st.info(
    "However, some graphs will help us understand and spot differences better."
)



st.write(" ")   

st.write("""
# Data Visualization

Let's make some graphs to help us understand our data better.
""")

def plot_violin(y2,i):
    plt.subplot(2,2,i)
    sns.violinplot(
        x='Iris name',
        y=y2,
        data=iris_df,
        palette={'Setosa': 'green', 'Versicolor': 'red', 'Virginica': 'blue'}
    )
    sns.despine(offset=10, trim=True)

plt.figure(figsize=(17,12))
i = 1

for measurement in iris_df.columns[:-2]:
    plot_violin(measurement,i)
    i += 1

st.pyplot(plt.gcf())

st.info("From the above violin plots we can notice high density of the length and width of setosa species, especialy for sepal length, petal length and petal width. Also we can observe that the mean values and the interquartile range for the petal measurements are easily distinguish, althought the values of virginica species are more spreaded.")

st.write(" ")
st.markdown("<span style='color:green'>**Lets produce a heatmap to find out the correlations between the measurements**</span>", unsafe_allow_html=True)


fig, axes = plt.subplots(figsize=(7,7))
sns.heatmap(iris_df.iloc[:,:4].corr(), annot = True, cbar=False)
axes.tick_params(labelrotation=45)
plt.title('Correlation heatmap', fontsize = 15)

st.pyplot(fig)

st.info("The heatmap above shows the correlation between the measurements. The strongest correlation is between petal length and petal width, with a value of 0.96. The weakest correlation is between petal width and sepal width, with a value of -0.37.")

