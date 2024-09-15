# ArabicTagger 
<a href='https://pypi.org/project/ArabicTagger/' target="_blank"><img alt='Python' src='https://img.shields.io/badge/Python_3.x-100000?style=plastic&logo=Python&logoColor=000000&labelColor=C7C7C7&color=73FFBB'/></a><a href='https://pypi.org/project/ArabicTagger/' target="_blank"><img alt='TensorFlow' src='https://img.shields.io/badge/TensorFlow_2.10.0-100000?style=flat&logo=TensorFlow&logoColor=FF7B2E&labelColor=D0CFCE&color=73FFBB'/></a><a href='https://pypi.org/project/ArabicTagger/' target="_blank"><img alt='Keras' src='https://img.shields.io/badge/Keras_2.10.0-100000?style=flat&logo=Keras&logoColor=FF0B0B&labelColor=D0CFCE&color=73FFBB'/></a>
<a href='https://www.kaggle.com/code/eslamtf/arabictagger' target="_blank"><img alt='Kaggle' src='https://img.shields.io/badge/Kaggle-100000?style=plastic&logo=Kaggle&logoColor=00BBFF&labelColor=D0CFCE&color=73FFBB'/></a>
<p style = "font-family:Cursive">
ArabicTagger is a Python package that has the following components:- 
<br>
1- a CRF layer implemented in Keras
<br>    
2- a BI-LSTM + CRF model implemneted in keras
<br>
3- Build and train your own Arabic NER models using pre-existing models with minimal lines of code
 and only the desired tags
</p>
<br>

## installation
```[python]
# ArabicTagger is still in its beta version   
# it's recommended to install it inside its own environment
pip install ArabicTagger
```
## test
```[python] 
from ArabicTagger import Tagger,NER,CRF
tagger = Tagger()
tagger.intialize_models()
inputs = [['السلام', 'عليكم', 'كم', 'سعر', 'الخلاط'],
         ['ما', 'هي', 'مواصفات', 'البوتجاز', 'الي', 'في', 'الصورة']]
tags =  [['DEVICE', 'O', 'O', 'O', 'O'],
         ['O', 'O','O', 'DEVICE', 'O', 'O', 'O']]
# define udt
user_defined_tags = ['DEVICE']
train1, _ = tagger.get_data(inputs, 7, tags, user_defined_tags)
X1,Y1 = train1
model = NER(20, 13, 7, 300, udt = [13])
model.compile(optimizer=tf.keras.optimizers.Adam(0.05))
model.fit([X1,Y1], Y1, epochs = 4 , batch_size = len(X1))
```
### CRF

<p style = "font-family:Cursive">
CRF is a Keras layer that has been created using subclassing, the call of the layer takes a list or tuple of the inputs and the output of shape (n+2, m) (n+2,) respectively another optional parameter is return_loss which is set to True by default. If return_loss is set to false the loss will be added to the final loss at the output layer of the model.
</p>
<br>

###  NER

<p style = "font-family:Cursive">
NER is the BI-LSTM + CRF model, this model goes beyond being a simple model but it has additional metrics defined inside it like udt_accuracy which calculates the total accuracy based on the user-defined tags this will be clear in the Tagger section 
</p>
<br>

### Tagger

<p style = "font-family:Cursive">
The Tagger module is a valuable asset for NLP tasks, particularly Named Entity Recognition (NER). It empowers users to create custom NER models by simply annotating their data. This annotation process involves labeling specific words or phrases as interest entities (e.g., "DEVICE").
<br>
    
<b>The Challenge of Limited Data</b>
    
<br>
Training an effective NER model often requires a substantial amount of annotated data. However, when dealing with limited datasets, especially those with a single dominant tag, achieving high accuracy can be challenging. This is because the model struggles to identify underlying patterns or structures within the data.
<br>
 
<b>Tagger's Solution: Tag Expansion</b>
    
<br>
The Tagger module addresses this limitation by generating additional tags using pre-trained models. This process, known as tag expansion, enriches the training data and helps the model discover hidden patterns.

<b> Introducing Our Pre-trained Models </b>
<br>
We offer two pre-trained models to facilitate tag expansion:
<br>
let's say you are building a NLP model that extracts the Device name from customers reviews 'Arabic Text!' in order to do this you need anotated data where each word to be Device or not will be somthing like this : 
['السلام', 'عليكم', 'كم', 'سعر', 'الخلاط'] and will be anotated like this ['DEVICE', 'O', 'O', 'O', 'O'].
<br>
if you tried different models to predict the outputs correctly,you will get very low accuracy that's because you don't have much training data and there is only one tag which make it harder for models to find some kind of structure.
<br>
Tagger module will enable you to generate more tags around your tag using pre-trained models, so that the model can capture some structure behind the data.currently we present two models :-
<br>
1 - CRF_model_1 (Part-of-speech model has 12 tags) 
<br>
the following data sets has been combined then splited for training :-
<br>
- <a href = "https://huggingface.co/datasets/QCRI/arabic_pos_dialect">arabic_pos_dialect(egy)</a>
<br>
- <a href = "https://universaldependencies.org/#language-">Arabic Data in universaldependencies</a>
    
<br>
    
Tag | Description | Example
----|-------------|--------
V   | Verb        | فعل (fi'l) - to do, to make
ADJ | Adjective   | صفة (sifa) - describing word
PART| Particle    | حرف (harf) - small word with grammatical function
PRON| Pronoun     | ضمير (ḍamīr) - word used instead of a noun
NUM | number      | رقم (raqm) - numeral
PREP|Preposition  | حرف جر (harf jar) - word used before a noun to show relationship
PUNC|punctuation  | علامة ترقيم (ʿalamāt tarqīm) - punctuation mark
DET | Determiner  | أل (al) - definite article
O   | object      | Outside of any named entity
ADV | Adverb      | ظرف (ẓarf) - word that modifies a verb, adjective, or adverb
CONJ| Conjunction | حرف عطف (harf ʿaṭf) - word that connects words or sentences
NOUN| Noun        | اسم (ism) - word that names a person, place, thing, or idea

2- CRF_model_2 (Named entity model has 7 tags)

| Tag     | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| I-LOC   | Inside a location entity, such as a country, city, or landmark.             |
| B-LOC   | Beginning of a location entity.                                             |
| I-PER   | Inside a person entity, such as a first name or last name.                  |
| B-PERS  | Beginning of a person entity.                                               |
| I-ORG   | Inside an organization entity, such as a company or institution.            |
| B-ORG   | Beginning of an organization entity.                                        |
| O       | Outside of any named entity, not part of any location, person, or organization. |
 
the following data set has been sampled then splited for training :-
<br>
- <a href = "https://sourceforge.net/projects/kalimat/files/kalimat/Corpus_Name_Entity_Recognition/">KALIMAT a Multipurpose Arabic Corpus(NER)</a>
    
</p>
<br>
<p style = "font-family:Cursive">Evaluation Results for Food Extraction

The proposed method was evaluated on a dataset of 600 food extraction examples. Using the first model, we achieved the following results:
<br>
Training: Accuracy: 0.9797, udt_accuracy: 0.9589
<br>
Testing: Accuracy: 0.9429, udt_accuracy: 0.9291
<br>
When using the second model, the results were:
<br>
Training: Accuracy: 0.9596, udt_accuracy: 0.9611
<br>
Testing: Accuracy: 0.9526, udt_accuracy: 0.9223
<br>
Overall, both models demonstrated strong performance in extracting food-related entities from the given dataset.
<br>
</p>


### another example has been provided in kaggle notebook to show how to train model to tag a stop word

### References

<br>
 1- https://aclanthology.org/J96-1002.pdf
 
 <br>

 2- https://www.bing.com/ck/a?!&&p=8a8829c85c466b73JmltdHM9MTcyNjM1ODQwMCZpZ3VpZD0xMDdhN2FmMS03MTMyLTY1MTYtMzNhMS02OTA5NzA0ZTY0OGMmaW5zaWQ9NTIwNA&ptn=3&ver=2&hsh=3&fclid=107a7af1-7132-6516-33a1-6909704e648c&psq=Log-linear+models++and+conditional+random+fields++Charles+Elkan&u=a1aHR0cHM6Ly9jc2V3ZWIudWNzZC5lZHUvfmVsa2FuLzI1MEJmYWxsMjAwNy9sb2dsaW5lYXIucGRm&ntb=1