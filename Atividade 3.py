import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT'] 
features = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
dados = pd.read_csv('hcvdatclear.csv',
                    names = names)

df = dados['Age']
df.sort_values(ascending=True)
print(dados.head())
# Amplitude dos dados = Valor maior dos registros - menor valor
classes = 5
print(df.max()) #valor máximo
print(df.min()) #valor mínimo
at = math.ceil((df.max() - df.min()) / classes)
print(at) #amplitude
frequencias = []

# Menor valor da série
menor = round(df.min(),1)

# Menor valor somado a amplitude
menor_amp = round(menor+at,1)

valor = menor
while valor < df.max():
    frequencias.append('{} - {}'.format(round(valor,1),round(valor+at,1)))
    valor += at
    
freq_abs = pd.qcut(df,len(frequencias),labels=frequencias) # Discretização dos valores em k faixas, rotuladas pela lista criada anteriormente
print(pd.value_counts(freq_abs))

plt.title('Idades de um grupo') 
plt.xlabel('Idade')
plt.ylabel('Frequência Absoluta')
plt.hist(dados["Age"], [19,37,45,56,67,79], rwidth=0.9)
plt.show()
plt.scatter(dados["Category"],dados["Age"])
plt.show()