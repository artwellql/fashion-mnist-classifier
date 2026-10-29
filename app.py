import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Fashion MNIST", layout="wide")

st.title("Класифікація одягу")

CLASSES = [
    'T-shirt/top (Футболка/топ)',
    'Trouser (Штани)',
    'Pullover (Светр)',
    'Dress (Сукня)',
    'Coat (Пальто)',
    'Sandal (Сандалі)',
    'Shirt (Сорочка)',
    'Sneaker (Кросівки)',
    'Bag (Сумка)',
    'Ankle boot (Черевики)'
]

# 1. Вибір моделі
st.subheader("1. Вибір моделі")
model_type = st.radio(
    "Оберіть архітектуру:",
    ("Згорткова нейромережа (Custom CNN)", "Модель на основі VGG16 (Fine-Tuning)")
)

is_vgg = "VGG16" in model_type
model_path = "vgg16_model.keras" if is_vgg else "custom_cnn_model.keras"
plot_path = "vgg16_history.png" if is_vgg else "cnn_history.png"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

model = load_model(model_path)

# 2. Виведення графіків точності та втрат
st.subheader("2. Графіки функції втрат і точності моделі")
if os.path.exists(plot_path):
    st.image(plot_path, caption=f"Історія навчання: {model_type}", use_container_width=True)
else:
    st.warning(f"Файл {plot_path} не знайдено в папці проекту. Збережіть графік із ноутбука під цією назвою.")

st.divider()

# 3. Завантаження та обробка зображення
st.subheader("3. Класифікація зображення")
file = st.file_uploader("Виберіть файл зображення", type=["png", "jpg", "jpeg"])

def preprocess_image(image, is_vgg_model):
    img_gray = image.convert('L')
    if is_vgg_model:
        img_resized = img_gray.resize((32, 32))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.stack((img_array,) * 3, axis=-1)
        return np.expand_dims(img_array, axis=0)
    else:
        img_resized = img_gray.resize((28, 28))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=(0, -1))

if file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(file)
    
    with col1:
        st.write("**Вхідне зображення:**")
        st.image(image, width=220)
        invert = st.checkbox("Інвертувати кольори (якщо тло біле)", value=True)
        
    image_to_predict = ImageOps.invert(image.convert('RGB')) if invert else image

    if st.button("Класифікувати"):
        if model is None:
            st.error(f"Помилка: файл моделі {model_path} не знайдено.")
        else:
            tensor = preprocess_image(image_to_predict, is_vgg)
            preds = model.predict(tensor)[0]
            pred_idx = int(np.argmax(preds))
            
            with col2:
                st.write("**Результати:**")
                st.write(f"**Передбачений клас:** {CLASSES[pred_idx]}")
                st.write(f"**Ймовірність:** {preds[pred_idx] * 100:.2f}%")
                
                st.write("**Ймовірності для всіх класів:**")
                probs_data = {CLASSES[i]: float(preds[i]) for i in range(10)}
                st.bar_chart(probs_data)