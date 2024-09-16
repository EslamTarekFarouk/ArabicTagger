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
 2- https://cseweb.ucsd.edu/~elkan/250Bfall2007/loglinear.pdf

 <br>

 # Arabic Text Matcher

**TextMatcher** is a module designed to address a common problem in
Arabic NLP:  
Suppose you’re building a model to handle customer orders and process
transactions automatically. You've implemented a Named Entity
Recognition (NER) model to extract device names in Arabic, such as
(<span dir="rtl">غسالة, تلفاز, شاشة, مروحة</span>). After identifying
these devices, you need to match them with entries in your database to
check stock availability.

However, issues arise when there's a spelling variation. For instance, a
customer might type "<span dir="rtl">غساله</span>," but in your
database, the device is listed as "<span dir="rtl">غسالة</span>." A
standard search would fail to match these two, even though they
represent the same item.

This issue is known as **orthographic** or **spelling variation** and is
a challenge for information retrieval in Arabic. Many approaches exist
to address this, and in the **ArabicTagger** package, I approached the
problem with a learnable weighted similarity. This method reduces the
number of parameters and shortens training time while improving matching
accuracy across spelling variations.

Let’s define a set of tuples each tuple has two words the first is the
word normalized one and the second word is the other variant e.x

K = <span dir="rtl"></span>{(“ <span dir="rtl">غسالة</span>”,”
<span dir="rtl">غساله</span>”),(“<span dir="rtl">مروحة</span>”,”<span dir="rtl">المروحه</span>”),…….}

And another set of tuples each tuple has two words that are dissimilar
to each other e.x

J = {(“
<span dir="rtl">غسالة</span>”,”<span dir="rtl">مروحة</span>”),(“<span dir="rtl">مروحة</span>”,”<span dir="rtl">تلاجة</span>”),…….}

Let S<sub>k</sub> be the similarity between two elements in the set K,
Sj be the similarity between the elements in the set J such that:

$`S_{k} = \frac{\sum_{i = 1}^{m}{W_{i}^{2}{\ I}_{ik}}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}}\ \ `$

$`S_{j} = \ \frac{\sum_{i = 1}^{m}{W_{i}^{2}{\ I}_{ij}}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}}\ \ \ \ `$

Where m is the vector length, W<sub>i</sub> is a learnable parameter for
each position in that vector.

We want to maximize the average similarity of pairs in the set K and
simultaneously minimize the average similarity of pairs in the set J, we
also want the weights W<sub>i</sub> to sum up to 1 (this constraint can
be relaxed) the problem can be formulated as follows.

$`\max_{W_{i}}{\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}S_{k} - \ \frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}S_{j}}\ \ `$

$`s.t\ \ \ \ \ \ \ \sum_{i = 1}^{m}{W_{i} = 1}`$

This problem can be solved using the Lagrange multipliers method as
follows :

$`L\left( W_{i},\ \ \lambda \right) = \ \frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}S_{k} - \ \frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}S_{j} - \lambda\ (\sum_{i = 1}^{m}{W_{i} - 1})`$.

$`L\left( W_{i},\ \ \lambda \right) = \ \frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}\frac{\sum_{i = 1}^{m}{W_{i}^{2}{\ I}_{ik}}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}} - \ \frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{\sum_{i = 1}^{m}{W_{i}^{2}{\ I}_{ij}}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}}\ } - \lambda\ (\sum_{i = 1}^{m}{W_{i} - 1})`$.

$`\nabla\ L = \ \begin{bmatrix}
\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{2\ W_{i}\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{{\ 2\ W}_{i}\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}}\  - \ \lambda\ \ } \\
\sum_{i = 1}^{m}{W_{i} - 1}
\end{bmatrix} = \ \begin{bmatrix}
0 \\
0
\end{bmatrix}\ `$

$`2\ W_{i}\ (\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\  = \ \lambda\ }`$

$`W_{i}\  = \ \frac{\lambda}{2}{(\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\ \ }}^{- 1}`$
(1)

$`{\sum_{i = \ 1}^{m}W}_{i} = \frac{\lambda}{2}\ \sum_{i = \ 1}^{m}{(\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\ }}^{- 1}`$
.

From the second constraint $`\sum_{i = 1}^{m}{W_{i} = 1}`$ then,

$`1 = \frac{\lambda}{2}\ \sum_{i = \ 1}^{m}{(\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\ }}^{- 1}`$
.

$`\lambda = 2\ {(\sum_{i = \ 1}^{m}{(\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\ }}^{- 1})}^{- 1} = \ 2N_{\mathcal{f}}`$

Where $`N_{\mathcal{f}}\ `$Is a normalization factor.

Substitute $`\lambda\ `$in equation (1) to get $`W_{i}`$ As follows :

$`W_{i}\  = \ N_{\mathcal{f}}{(\frac{1}{n_{1}}\ \sum_{k = 1}^{n_{1}}{\frac{{\ \ I}_{ik}\ I_{ik}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ik}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ik}^{\sim}}^{2}}}\  - \ }\frac{1}{n_{2}}\ \sum_{j = 1}^{n_{2}}{\frac{{\ \ I}_{ij}\ I_{ij}^{\sim}}{\sqrt{\sum_{i = 1}^{m}I_{ij}^{2}}\ \sqrt{\sum_{i = 1}^{m}{I_{ij}^{\sim}}^{2}}})\ \ }}^{- 1}`$

That simply means the model will estimate each weight. $`W_{i}`$ Based
on the inverse of the similarity of both sets K and J at that position
i.

This approach is similar to the attention mechanism, where the model
focuses on positions that lead to high similarity in set K and low
similarity in set J. The positions are defined using a specific strategy
in **TextMatcher** as follows:

{

'<span dir="rtl">ا': 0</span>,

'<span dir="rtl">ب': 1</span>,

'<span dir="rtl">ت': 2</span>,

'<span dir="rtl">ة': 3</span>,

'<span dir="rtl">ث': 4</span>,

'<span dir="rtl">ج': 5</span>,

'<span dir="rtl">ح': 6</span>,

...

'<span dir="rtl">اه': 61</span>,

'<span dir="rtl">او': 62</span>,

'<span dir="rtl">اي': 63</span>,

'<span dir="rtl">اء': 64</span>,

'<span dir="rtl">اآ': 65</span>,

'<span dir="rtl">اأ': 66</span>,

'<span dir="rtl">اؤ': 67</span>,

'<span dir="rtl">اإ': 68</span>,

...

}

The dictionary contains 1,260 entries. We initialize an empty vector of
size 1,260, then loop through each word's 1-grams and 2-grams. For each
match, we update the corresponding vector position using the following
formula:

Vector\[i\] = Vector\[i\] + 1 + func(i)

Here, func(i) is a position-encoding function that adjusts the value
based on the position of the character or 2-gram within the word.

In future versions, I hope to add other methods to deal with this
problem

For more info about how to use **TextMatcher** see Kaggle notebook at
GitHub.
