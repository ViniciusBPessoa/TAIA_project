import sys
import os
script_dir = os.getcwd() # coleata o diretorio do projeto atual
sys.path.append(script_dir)

import pandas as pd
import numpy as np
import nltk
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models, layers
from keras.models import clone_model
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

from copy import deepcopy

def create_MLP_model(input_dim, output_dim, dense_layers):
    model = models.Sequential()
    
    # Redimensionar os dados de entrada
    model.add(layers.Reshape((input_dim,), input_shape=(input_dim, 1)))
    
    model.add(layers.Dense(dense_layers[0], activation='relu'))

    for neurons in dense_layers[1:]:
        model.add(layers.Dense(neurons, activation='relu'))

    model.add(layers.Dense(output_dim, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

class Genetic_algoritm:
    def __init__(self, tamanho_populacao : int, numero_prog : int, alpha : float, MLPdim : tuple[3], data : tuple[4]) -> None:
        self.tamanho_populacao = tamanho_populacao
        self.numero_prog       = numero_prog
        self.alpha             = alpha
        self.MLPdim            = MLPdim
        self.populacao         = []

        self.X_train = data[0]
        self.Y_train = data[1]
        self.X_test  = data[2]
        self.Y_test  = data[3]

        self.iniciar_gen()
    
    def iniciar_gen(self) -> None:
        for x in range(self.tamanho_populacao): self.populacao.append([create_MLP_model(self.MLPdim[0], self.MLPdim[1], self.MLPdim[2]), 0])

    def pontuador(self, resposta_prevista : int, resposta_real : int) -> int:
        return 1 if resposta_prevista == resposta_real else 0
    
    def avaliador(self, individuo) -> int:
        pontuacao = 0
        resposta_prevista = individuo[0].predict(self.X_train)
        resposta_prevista = np.argmax(resposta_prevista, axis=1)
        print(len(resposta_prevista))
        
        for posicao, pontuacao in enumerate(resposta_prevista): pontuacao += self.pontuador(pontuacao, self.Y_train[posicao])

        return pontuacao
    
    def avaliador_geracional(self) -> None:
        for index, individuo in enumerate(self.populacao): self.populacao[index][1] = self.avaliador(self.populacao[index])
    
    def seleciona_progenitores(self) -> list:
        redes_branch_ordenada = sorted(self.populacao, key=lambda item: item[1], reverse=True)
        return redes_branch_ordenada[:self.numero_prog]
    
    def gerador_descendentes(self) -> None:
        lista_pais   = self.seleciona_progenitores()
        descendentes = self.tamanho_populacao - self.numero_prog
        prox_gen     = []
            
        for x in range(descendentes): prox_gen.append(lista_pais[x % len(lista_pais)])
        for indiv in range(len(prox_gen)):
            prox_gen[indiv][0] = self.mutacao(prox_gen[indiv][0])
            
        self.populacao = prox_gen.extend(lista_pais)
    
    def mutacao(self, individuo):
        rede_mutada = deepcopy(individuo)

        for layer in rede_mutada.layers:
            if len(layer.get_weights()) > 0:
                pesos, biases = layer.get_weights()
                novos_pesos   = pesos + self.alpha * np.random.randn(*pesos.shape)
                
                layer.set_weights([novos_pesos, biases])

        return rede_mutada
    
    def estrategia_evolutiva(self, n_geracoes : int) -> list:
        for x in range(n_geracoes):
            self.avaliador_geracional()
            self.gerador_descendentes()
            
        return self.populacao[:1]