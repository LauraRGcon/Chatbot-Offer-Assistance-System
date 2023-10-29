import pandas as pd
import sqlite3
import json
import numpy as np
from scipy.special import softmax
import re
import string
import spacy
import tensorflow as tf
from transformers import BertTokenizerFast,TFBertForTokenClassification
from word2number import w2n

# variables de etiquetas y procesado

label_mapping = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6, 'july': 7, 'august': 8, 'september': 9,
                 'october': 10, 'november': 11, 'december': 12}

id2label ={
    0: 'O',
    1: 'adults',
    2: 'children',
    3: 'month',
    4: 'days',
    5: 'special_date'
}

model_type='dslim/bert-base-NER'


def number_conversion(entrada: str, default: int) -> int:
    try:
        numero = w2n.word_to_num(entrada) if isinstance(entrada, str) else entrada
        return numero
    except ValueError:
        return default


# Funciones y clases

def create_database(client):
 # df=pd.read_csv('../Data/offers_data.csv') # datos reales en local
  df=pd.read_csv('../Data/df_dummy.csv') # opción para github - dummy data

  df['startDate'] = pd.to_datetime(df['startDate'])

  df.to_sql('offers', client, index=False, if_exists='replace')


class Chatbot:
    def __init__(self):
        self.model = TFBertForTokenClassification.from_pretrained("../models/final_model", num_labels=6)
        self.tokenizer =  BertTokenizerFast.from_pretrained(model_type)

        self.reset()

    def reset(self):
        self.entities =  {'adults': None,
                           'children': None,
                           'days': None,
                           'month': None,
                           'special_date': None}


    def preprocess_text(self,texto, max_len=128):
        tokens = self.tokenizer.encode_plus(texto.lower(), max_length=max_len, truncation=True,  
                                            padding='max_length', add_special_tokens=True, return_tensors='tf')

        return {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']},

    def extract_entities(self, text):
        # limpieza de texto, creación de lista de tokens, extraccion de entidades
        input_process = self.preprocess_text(text)
        token_lists= self.tokenizer.decode(input_process[0]['input_ids'][0]).split()
        prediction = self.model.predict(input_process)

        #Obtención de etiquetas
        logits = prediction.logits if hasattr(prediction, 'logits') else prediction['logits']
        probs = softmax(logits, axis=-1)
        predictions = np.argmax(probs, axis=-1)

        # Mapeo de etiquetas con descriptivos (claves de entities)
        predicted_labels = [[id2label[id] for id in sentence] for sentence in predictions]

        # actualizacion de diccionario de entidades
        for i,label in enumerate(predicted_labels[0]):
          if label!='O':
            self.entities[label] =token_lists[i] if token_lists[i] != '[SEP]' else token_lists[i-1] 
        return self.entities

    def fetch_data(self, entities):
        # Realiza la consulta a BBDD SQL

        base_query='select * from offers where '

        entities['adults']=number_conversion(entities['adults'],2)

        adults_query= f" adults = {entities['adults']}"


        if entities['children'] is None or entities['children']==0:
            children_query=''' and childrenAges='[]' '''
        else:
            children_query=''' and childrenAges!='[]' '''

        extra_info=''
        if entities['days']:

            entities['days']=number_conversion(entities['days'],5)

            extra_info+= f" and lenght>={entities['days']}"

        if entities['month']:
            month_num=label_mapping.get(entities['month'],0)
            extra_info+= f" and strftime('%m', startDate)='{'0'+ str(month_num) if month_num<10 else month_num}'"

        elif  entities['special_date']:
            if entities['special_date']=='summer': #summer
                extra_info+= f" and strftime('%m', startDate) BETWEEN '06' AND '08'"
            elif entities['special_date'] in ['christmas','boxing day']: #Christmas or Boxing Day
                extra_info+= f" and strftime('%m', startDate)='12'"
            elif entities['special_date']=='new year': #New Year
                extra_info+= f" and strftime('%m', startDate)='01'"
            elif entities['special_date']=='valentine': #Valentine
                extra_info+= f" and strftime('%m', startDate)='11'"
            elif entities['special_date']=='halloween': #Halloween
                extra_info+= f" and strftime('%m', startDate)='02'"
            else:
                extra_info+=''

        query=base_query+adults_query+children_query+extra_info

        return pd.read_sql_query(query, con)


    def offer_selection(self,df):

        df=df.sort_values(['totalDiscount'],ascending=[False]).head(1)

        if len(df)>0:
            row = df.iloc[0]

            if row['type'] == 'accommodationProductDiscount':
                print(f"You're in luck! The room {row['room']} at the hotel {row['hotelName']} is discounted by {row['totalDiscount']}% for a stay from {row['startDate']} to {row['endDate']} for just {row['finalCost']}$.")
            else:
                print( f"You're in luck! The room {row['room']} at the hotel {row['hotelName']} for a stay from {row['startDate'][:10]} to {row['endDate']} is only {row['roomCosts']}$. Also, for these dates, there are additional offers {row['offerNames']} which combine the following discounts: {row['marketingText']} discount, so you could save up to {row['totalDiscount']}% by booking the discounted products, for only {row['finalCost']}$.")

        else:
            print('I am sorry! I have not been able to find an offer that meets your requirements, please try again.')

    def chat(self):
        print('*****'*20)
        print("Hello! Are you looking for the best booking price? Please tell me what are you interested in? \n You can say 'reset' at any time to perform another search.")

        while True:
            text = input()
            if text.lower() == "exit":
                print("Goodbye!")
                break
            elif text.lower() == "reset":
                self.reset()
                print("Request restarted, how can I help you now?")
                continue


            entities = self.extract_entities(text)
            if (not entities["adults"]):
                print("Please, specify the number of guests.")
                text = input()  # Esperamos la respuesta del usuario
                try:
                    num_adults = int(text)
                    self.entities["adults"] = num_adults
                except ValueError:
                    print("That doesn't look like a valid number. Please try again.")
                    continue


            if (not entities["month"]) and (not entities["special_date"]):
                print("Please, specify the month you want to make the reservation.")
                text = input()  # Esperamos la respuesta del usuario
                month_pattern = re.compile(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", re.IGNORECASE)
                match = month_pattern.search(text)

                if match:
                    self.entities["month"] = match.group(0).lower()
                else:
                    print("Sorry, I could not identify the month.")
                    continue

            results = self.fetch_data(entities)

            self.offer_selection(results)

            print('\n')
            print("Can I help you with anything else? (type 'exit' to leave)")
            self.reset()

if __name__ == "__main__":


	con = sqlite3.connect("BBDD_SQL") # creación de BBDD sintética
	create_database(con)              # Carga de datos en la BBDD
	bot = Chatbot()                   # Instanciamiento del bot
	bot.chat()                        # Inicio de la comunicación
	con.close()                       # cierre del cliente SQL