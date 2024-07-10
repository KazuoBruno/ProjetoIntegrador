import pandas as pd

teste = pd.read_csv('datafinal.csv')

x = teste.drop('Grau_Acidente', axis=1).copy()
print(teste)

