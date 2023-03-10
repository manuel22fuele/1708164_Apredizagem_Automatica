import matplotlib.pyplot as plt
import pickle as p1
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequality-white.csv",
sep=";") -- ou -- sep=",")

## Actividade 2
## Indique a lógica e o procedimento de cada linha do código em baixo.

train_data = data[:1000] ##  Este comando cria uma nova variável chamada "train_data" e
# atribui a ela uma cópia dos primeiros 1000 elementos da variável "data".
data_X = train_data.iloc[:,0:11] ## Este comando atribui a variável "data_X" um subconjunto
# dos dados contidos no DataFrame "train_data". Mais especificamente, ele seleciona todas as
# linhas do DataFrame "train_data" e as colunas da posição 0 a 10 (ou seja, as colunas de índice 0 a 10)
# e atribui esses dados à variável "data_X".
data_Y = train_data.iloc[:,11:12] ## Este comando cria um novo objeto chamado "data_Y" e atribui a
# ele uma parte do objeto "train_data". A parte selecionada é obtida usando o método "iloc()",
# que permite selecionar partes do objeto pelos seus índices numéricos. Neste caso, a parte
# selecionada inclui todas as linhas do objeto "train_data" e apenas a coluna de índice 11.
#print(train_data.columns)
print(data_X) ## Imprime no ecrã os dados da váriável data_X.
print(data_Y) ## Imprime no ecrã os dados da váriável data_Y.


## Actividade 3
## Indique a lógica e o procedimento de cada linha do código em baixo.

regr = linear_model.LinearRegression() ## Esse comando cria um objeto chamado regr da classe
# LinearRegression no módulo linear_model da biblioteca scikit-learn do Python.
preditor_linear_model = regr.fit(data_X, data_Y) ## Este comando cria um modelo de regressão
# linear chamado "preditor_linear_model" usando a função "fit()" da biblioteca scikit-learn do Python,
# que ajusta o modelo aos dados de entrada "data_X" e "data_Y".
# Onde:
# "data_X" é uma matriz numpy que contém as variáveis independentes (também conhecidas como
# características ou atributos) usadas para prever a variável dependente "data_Y".
# "data_Y" é um vetor numpy que contém a variável dependente que se deseja prever.
# "regr" é uma instância de um objeto regressor linear da classe "LinearRegression()"
# da biblioteca scikit-learn.
preditor_Pickle = open('../white-wine_quality_predictor', 'wb')# Este comando cria um objeto
# de arquivo com o nome "preditor_Pickle" e o abre no modo de gravação de bytes ("wb") no
# diretório pai (../) do diretório atual. Este objeto de arquivo será usado para gravar
# dados em um arquivo binário.
# O nome do arquivo a ser criado ou modificado é "white-wine_quality_predictor". É provável
# que este arquivo seja usado para armazenar um modelo de aprendizado de máquina treinado em
# prever a qualidade do vinho branco com base em dados de entrada. A extensão do arquivo não é
# especificada, mas é comum usar a extensão ".pkl" para arquivos de modelo serializados usando o
# módulo "pickle" do Python.
print("white-wine_quality_predictor") # O comando "print("white-wine_quality_predictor")" imprime a
# string "white-wine_quality_predictor" na saída do console, que é a janela onde o código é executado.
# A função print() é uma função nativa do Python que permite exibir informações na tela. Neste caso,
# a string "white-wine_quality_predictor" será exibida na tela quando o código for executado.
p1.dump(preditor_linear_model, preditor_Pickle) # Este comando provavelmente está usando a biblioteca
# "pickle" do Python para salvar o objeto "preditor_linear_model" em um arquivo chamado "preditor_Pickle".

import.... + data = ## Este import repete todos os imports feito anteriormente, afim de não ser repetitivo.
evaluation_data=data[1001:] # Este comando está a criar uma variável chamada "evaluation_data" que
# receberá uma parte dos dados contidos na variável "data". A parte selecionada começa a partir
# do índice 1001 e vai até o final da sequência, pois nenhum índice final foi especificado após
# o sinal de dois pontos ":". Portanto, a variável "evaluation_data" conterá todos os elementos
# de "data" a partir do índice 1001 até o final da sequência.
data_X=evaluation_data.iloc[:,0:11] # Este comando atribui a variável "data_X" um subconjunto
# dos dados contidos no DataFrame "evaluation_data". Mais especificamente, ele seleciona todas as
# linhas do DataFrame "evaluation_data" e as colunas da posição 0 a 10 (ou seja, as colunas de índice
# 0 a 10)
# e atribui esses dados à variável "data_X".
data_Y=evaluation_data.iloc[:,11:12] # Este comando cria um novo objeto chamado "data_Y" e atribui a
# ele uma parte do objeto "evaluation_data". A parte selecionada é obtida usando o método "iloc()",
# que permite selecionar partes do objeto pelos seus índices numéricos. Neste caso, a parte
# selecionada inclui todas as linhas do objeto "evaluation_data" e apenas a coluna de índice 11.
print(type(evaluation_data)) # Indica que tipo de dados é a variável "evaluation_data".
print(type(data_X)) # Indica que tipo de dados é a variável "data_X".
loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb')) # O comando
# "loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb'))"
# carrega um modelo de aprendizado de máquina salvo em disco usando a biblioteca
# "pickle" do Python.
print("Coefficients: \n", loaded_model.coef_) ## Este comando imprime na tela os coeficientes do
# modelo de aprendizado de máquina carregado (loaded_model). A função print() é usada para exibir
# na tela uma mensagem formatada, que é composta por duas partes:
#   A primeira parte é a string "Coefficients: \n", que é o texto que será exibido na tela. O
#   caractere \n é usado para inserir uma quebra de linha, o que significa que o texto seguinte
#   será exibido em uma nova linha.
#   A segunda parte é a variável loaded_model.coef_, que contém os coeficientes do modelo de
#   aprendizado de máquina carregado. Os coeficientes são uma lista ou array de valores numéricos
#   que representam os pesos atribuídos a cada uma das variáveis de entrada do modelo.
# Portanto, a mensagem completa exibida na tela pelo comando
# print("Coefficients: \n", loaded_model.coef_) será "Coefficients:",
# seguida de uma quebra de linha, e em seguida, a lista de coeficientes do modelo.
y_pred=loaded_model.predict(data_X) # O comando "y_pred=loaded_model.predict(data_X)" é usado
# para fazer previsões em um conjunto de dados de entrada "data_X" usando um modelo de aprendizado
# de máquina previamente treinado chamado "loaded_model" e armazenar as previsões resultantes na
# variável "y_pred".
z_pred = y_pred - data_Y # Este comando realiza uma operação de subtração entre dois arrays:
# "y_pred" e "data_Y". O resultado dessa subtração é atribuído a uma variável "z_pred".
# Assumindo que "y_pred" e "data_Y" são arrays numpy ou similares, a subtração é realizada
# elemento por elemento. Isso significa que o primeiro elemento de "y_pred" é subtraído do
# primeiro elemento de "data_Y", o segundo elemento de "y_pred" é subtraído do segundo elemento
# de "data_Y" e assim por diante. O resultado dessa operação é armazenado na variável "z_pred".


## Actividade 4
## Indique a lógica e o procedimento de cada linha do código em baixo.

right=0 & wrong=0 & total=0 # Este comando atribui o valor '0' para as variáveis "right", "wrong" e "total".
for x in z_pred["quality"]: ## O comando é um laço de repetição que é usado para percorrer uma
# sequência de valores contidos em uma variável chamada "z_pred".
    z = int(x) # Este comando converte a variável "x" em um número inteiro e armazena o resultado
# na variável "z".
total = total+1 # Este comando adiciona 1 ao valor atual da variável "total" e atribui o
# novo valor resultante à variável "total".
if z==0: # Está condição diz-nos que se a variável z for igual a zero (0)
    right=right+1 # Adiciona-se 1 ao valor atual da variável "right" e atribui o novo valor resultante à variável "right".
else: # Caso seja um outro valor
    wrong=wrong+1 # Adiciona-se 1 ao valor atual da variável "wrong" e atribui o novo valor resultante à variável "wrong".

    print("accuraccy1= ",right/total,"accuraccy2= ",wrong/total) # Este comando imprime na tela
# duas medidas de precisão (accuracy) em um formato de string:

#       A primeira medida, "accuraccy1", é o número de respostas corretas dividido pelo total de
#       respostas (right/total).
#       A segunda medida, "accuraccy2", é o número de respostas incorretas dividido pelo total de
#       respostas (wrong/total).
# Ambas as medidas são separadas por uma vírgula e um espaço na string impressa na tela.
# Por exemplo, se right=20, wrong=5 e total=25, a saída seria:
#       accuraccy1= 0.8 accuraccy2= 0.2


## Actividade 5
# Indique a lógica e o procedimento de cada linha do código em baixo.

import.... ## repete os imports feitos anteriormente.
data_x=input("introduza valores do wine\n") # Pede para o usuário inserir um texto.
data=data_x.split(";") # A variável "data" armazena o que está na variável "data_x", mas separado por ';'.
print(data) # Mostra na pela a informação contida na variável "data".
fmap_data = map(float, data) # O comando "fmap_data = map(float, data)" converte cada elemento
# de uma lista "data" para o tipo float usando a função "float()", e armazena o resultado em um
# objeto "map" chamado "fmap_data".
print(fmap_data) # Mostra na tela a informação contida na variavel "fmap_data".
flist_data = list(fmap_data) # Este comando cria uma nova lista chamada
# "flist_data" contendo os mesmos elementos da lista "fmap_data".
print(flist_data) # Mostra na tela a informação contida na variável "flist_data".
data1 = pd.read_csv("../VinhoB/Qualidade_vinho_B/winequalitywhite.csv",sep=";") # Este comando
# carrega um arquivo CSV (comma-separated values) contendo dados sobre a qualidade do
# vinho branco em um objeto Pandas DataFrame chamado "data1
data2 = data1.iloc[:0,:11] # Esse comando está utilizando o método iloc para selecionar um
# subconjunto dos dados contidos no objeto data1. Portanto, o comando cria um novo objeto
# DataFrame chamado data2 que contém as mesmas colunas que data1, mas sem nenhuma linha.
data_preparation = pd.DataFrame([flist_data],columns=list(data2)) # Este comando cria um
# objeto DataFrame do pandas chamado "data_preparation" a partir de uma lista unidimensional
# "flist_data". Os dados na lista são organizados em colunas usando os rótulos de coluna fornecidos
# na lista "list(data2)".
out=data2 # Atribui os dados contido na variável "data2" a variável "out"
for x in out: # indica que o bloco de código seguinte será executado para cada elemento da
# variável "out".
print(x,data_preparation[x].values) # Mostra na tela cada elemento "x" contido em "out"
# e o seu valor no DataFrame chamando "data_preparetion".
loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb')) # O comando
# "loaded_model = p1.load(open('../white-wine_quality_predictor', 'rb'))"
# carrega um modelo de aprendizado de máquina salvo em disco usando a biblioteca
# "pickle" do Python.
y_pred = loaded_model.predict(data_preparation)# O comando "y_pred=loaded_model.predict(data_preparation)" é usado
# para fazer previsões em um conjunto de dados de entrada "data_preparation" usando um modelo de aprendizado
# de máquina previamente treinado chamado "loaded_model" e armazenar as previsões resultantes na
# variável "y_pred".
print("wine quality",int(y_pred)) # Mostra na tela e converte para inteiro os elementos contidos na variável "y_pred".
