# Imports
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
import streamlit as st
import string

########################################################################################################################################

image = Image.open('web-g4bea507f7_1920.jpg')
st.image(image, caption='freepik', width=100, use_column_width='always')

st.title('Speak With Me!')
#########################################################################################################################################

# carrega modelo de nlp
modelo_nlp = spacy.load('modelo_nlp')


def clf1(texto):
    
    # padrão de elementos de caracteres
    pontuacoes = string.punctuation
    # carrega dicionario pt br
    pln = spacy.load("pt_core_news_sm")
    stop_words = STOP_WORDS
    
    def preprocess(texto):
        texto = texto.lower()
        documento = pln(texto)
        
        lista = []
        for token in documento:
            lista.append(token.lemma_)
        lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]  
        lista = ' '.join( [str(elemento) for elemento in lista if not elemento.isdigit()] )
        
        
        return lista

    previsao = modelo_nlp(preprocess(texto))
    df_proba_nlp = pd.DataFrame(previsao.cats, index=[0])
    
    return st.write(df_proba_nlp)


def one():
    
    st.title('Seja respondido por uma IA !')
    
    html_temp = """
                """
    st.markdown(html_temp)
    
    texto = st.text_area("Escreva o texto que o lead informou no diálogo",value="")
    
    df_proba_nlp = " "
    
    if st.button("Prever a necessidade de acordo ao texto digitado"):
        st.text('Resultados em uma tabela de probabilidades')
        df_proba_nlp = clf1(texto)

st.markdown("[**By Eric Oliveira**](https://www.linkedin.com/in/eric-oliveira-ds) ")

if __name__ == '__main__':
    one()
    
