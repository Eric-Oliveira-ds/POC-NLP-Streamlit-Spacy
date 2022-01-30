# Imports
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
from PIL import Image
import spacy
import spacy.cli
spacy.cli.download("pt_core_news_sm")
from spacy.lang.pt.stop_words import STOP_WORDS
import streamlit as st
import string

########################################################################################################################################
st.sidebar.subheader('About the app')
st.sidebar.text('Speak With Me!')
st.sidebar.info('Use the App and discover the interests of services for areas of technologies from texts.')
st.sidebar.subheader('Developer by Eric Oliveira - [**LinkedIn**](https://www.linkedin.com/in/eric-oliveira-ds) ')

image = Image.open('web-g4bea507f7_1920.jpg')
st.image(image, caption='freepik', width=100, use_column_width='always')

st.title('------------ Speak With Me! ------------')
#########################################################################################################################################
# carrega modelo de nlp
modelo_nlp = spacy.load('modelo_nlp')


def prevendo(texto):
    
    # padrão de elementos de caracteres
    pontuacoes = string.punctuation
    # remover textos redundantes(de, para, como) e deixar os com mais informação (perdeu, ganhou)
    stop_words = STOP_WORDS

    @st.cache
    def preprocess(texto):
        texto = texto.lower()
        # carrega dicionario pt br
        pln = spacy.load("pt_core_news_sm")
        # aplica tratamento no texto recebido pelo usuario
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

def main():
    
    st.title('Be answered by an AI !')
    
    html_temp = """
                """
    st.markdown(html_temp)
    
    texto = st.text_area("Write the text that the lead informed in field below !",value="")
    
    df_proba_nlp = " "
    
    if st.button("Predict the need"):
        st.text('Results in a table below:')
        df_proba_nlp = prevendo(texto)

if __name__ == '__main__':
    main()
