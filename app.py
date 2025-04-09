from joblib import load
import numpy as np
from keras.models import load_model
import streamlit as st


def get_image_class(target_class):
    image_path = ['img/setosa.jpg', 'img/versicolor.jpg', 'img/virginica.jpg']
    return image_path[target_class]


# data app for iris prediction (classification)

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
        st.image(get_image_class(np.argmax(classes_prob)))
        text = class_labels[np.argmax(classes_prob)]
    else:
        iris_class = model_knn.predict(
            np.array([[petal_length_input, petal_width_input]]))
        st.image(get_image_class(iris_class[0]))
        text = class_labels[iris_class[0]]

    st.html(
        f"""<p class="answer-color">{text}</p>""")

st.caption('IA pode cometer erros. Considere verificar informações importantes')
