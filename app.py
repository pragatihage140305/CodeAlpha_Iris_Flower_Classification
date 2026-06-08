import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(X, y)

st.title("🌸 Iris Flower Classifier")

sl = st.number_input("Sepal Length")
sw = st.number_input("Sepal Width")
pl = st.number_input("Petal Length")
pw = st.number_input("Petal Width")

if st.button("Predict"):
    prediction = model.predict([[sl, sw, pl, pw]])
    st.success(f"Predicted Flower: {iris.target_names[prediction][0]}")