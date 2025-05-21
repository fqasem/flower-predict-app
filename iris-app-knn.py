st.info("The next thing we are going to use and test the k-Nearest Neighbors model. ")

st.write(" ")
st.subheader("Training the k-Nearest Neighbors Model")
st.write("We are going to use and test the k-Nearest Neighbors model, and since out data does not seem **noisy** we can choose a small value of k. We will set the k to 3.")
st.write("Although we noticed that high correlaton between the petal width and length measurements, we will use all the mesurements available at the moment, and later check which gives the better accuracy.")
st.write("Furthermore keep in mind that KNN is calculating the euclidean distance between the point we want to predict and the nearest(s) training data point(s) (neighbor). To this end scaling (normalizing) the data before applying the alogirthm usually is a good approach. However in our case all the data use the same unit of measurement (cm) so this is not necessary.")


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# Source code credit for this section: https://www.kaggle.com/code/kostasmar/exploring-the-iris-data-set-scikit-learn

X = iris_df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)','petal width (cm)']]
y = iris_df['Iris name']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)