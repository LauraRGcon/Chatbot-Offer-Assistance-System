# Assistance System for Offer Matching between Hotels and Travel Agencies 

## Instalación

Para instalar las librerías necesarias: 


```python
pip install -r requirements.txt
```


Versión de python utilizada 3.10.12


## Estructura del proyecto:


+ Code: 
    1. Train Data Creation.ipynb : Notebook con el código de generación de datos de texto y etiquetas. 
    2. Train Models.ipynb : Código de limpieza de textos y pruebas de entrenamiento de algoritmos de deep learning.
    3. ChatBot.py: Código final del proyecto, gestión de interactuación con el usuario y sistema de recomendación.
    
+ Data:
    1. df_dummy.csv : Archivo con datos de ofertas anonimizadas
    2. train_data.pickle: Archivo generado con el código "Train Data Creation" que compone el conjunto de entrenamiento etiquetado para el procesado NLP.
    
+ Models:
    1. final_model: Modelo final seleccionado de fine tuning de transformers
    
+ Utils:
    1.sentences.py : Plantilla de frases para generación de dataset de entrenamiento NLP
  
## Activación del Chatbot


Dada la limitación de espacio del repositorio, el modelo final preentrenado se ha comprimido en varios archivos de 100MB. Es necesario realizar su descompresión antes de ejecutar el código chatbot.py.

Para inicializar el chat, dentro del path de la carpeta "Code" del proyecto, ejecutar en la consola de comandos:

```python
python chatbot.py
```