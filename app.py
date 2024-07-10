import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

df = pd.read_csv('datafinal.csv')

y = df['Grau_Acidente'].copy()
not_columns = ['Grau_Acidente']
X = df.drop(not_columns, axis=1).copy()
X = X.astype(np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=100)

inputs = tf.keras.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

st.title("Prevendo Grau do Acidente")
st.divider()

vLat = st.number_input("Digite a Latitude")
vLon = st.number_input("Digite a Longitude")
vMin = st.text_input("Digite a hora e o minuto")
vBai = st.text_input("Digite o bairro")
vMes = st.number_input("Digite o Mes")
vAno = st.number_input("Digite o Ano")

valores = [[]]

if st.button("Calcular"):

    valores[0].append(float(vLat))
    valores[0].append(float(vLon))
    vMinFin = (int(vMin[:2]) * 60) + int(vMin[4:])
    valores[0].append(int(vMinFin))
    bairros = ['BA_BRONX', 'BA_BROOKLYN', 'BA_MANHATTAN', 'BA_QUEENS', 'BA_STATEN ISLAND']	

    for i in range(len(bairros)):
        if vBai in bairros[i]:
            valores[0].append(1)
        else:
            valores[0].append(0)

    valores[0].append(int(vMes))
    valores[0].append(int(vAno))
    st.write(valores)
    valores = pd.DataFrame(valores, columns=['LATITUDE', 'LONGITUDE', 'Minutos_Total', 'BA_BRONX', 'BA_BROOKLYN',
       'BA_MANHATTAN', 'BA_QUEENS', 'BA_STATEN ISLAND', 'Start_Time_Month',
       'Start_Time_Year'])
    valores = valores.astype(np.float64)
    valores = scaler.transform(valores)
    #  Fazendo previsões
    predictions = model.predict(valores)

    # Exibindo previsões
    for i, prediction in enumerate(predictions):
        predicted_class = np.argmax(prediction)
        st.write(f"Classe Prevista: {predicted_class}")
