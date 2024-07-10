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

st.title("Prevendo tanana")
st.divider()

var1 = st.number_input("Digite um numero")

if var1 == 1:
    st.write("bruno eh legal")
