# data app for iris prediction (classification)
import streamlit as st
from keras.models import load_model
import numpy as np
from joblib import load

model_nn = load_model('models/model.keras')
model_knn = load('models/model-knn.pkl')

# assumindo este labelEncoder
class_labels = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']

# st.set_page_config(layout="wide")

st.title('Classificação de Flor :blue[Iris]')

petal_length_input = st.slider("Comprimento da pétala", 1.0, 6.9, (1.0+6.9)/2)
petal_width_input = st.slider("Espessura da pétala", 0.1, 2.5, (0.1+2.5)/2)

btn = st.button('Pergunte à IA')

model_opt = st.selectbox("Escolha o classificador", ["Rede Neural", "KNN"])

st.markdown("""
<style>
.big-font {
    font-size:300px !important;
}
.answer-color {
    color:yellow;
}
</style>
""", unsafe_allow_html=True)

if btn:
    if model_opt == "Rede Neural":
        classes_prob = model_nn.predict(
            np.array([[petal_length_input, petal_width_input]]))
        if (np.argmax(classes_prob) == 0):
            st.image('img/setosa.jpg')
        elif (np.argmax(classes_prob) == 1):
            st.image('img/versicolor.jpg')
        else:
            st.image('img/virginica.jpg')
        text = class_labels[np.argmax(classes_prob)]
    else:
        iris_class = model_knn.predict(
            np.array([[petal_length_input, petal_width_input]]))
        if (iris_class == 0):
            st.image('img/setosa.jpg')
        elif (iris_class == 1):
            st.image('img/versicolor.jpg')
        else:
            st.image('img/virginica.jpg')
        text = class_labels[iris_class[0]]
        print(iris_class[0])

    st.html(
        f"""<p class="answer-color">{text}</p>""")

st.caption('IA pode cometer erros. Considere verificar informações importantes')
