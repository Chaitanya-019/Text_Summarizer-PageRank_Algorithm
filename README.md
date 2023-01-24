# Text_Summarizer-PageRank_Algorithm

In this repository, we'll try to implement extractive Text Summarisation using  TextRank algorithm in python.

The steps followed are:

-> The first step would be to concatenate all the text contained in the articles

-> Then split the text into individual sentences

-> In the next step, we will find vector representation (word embeddings) for each and every sentence

-> Similarities between sentence vectors are then calculated and stored in a matrix

-> The similarity matrix is then converted into a graph, with sentences as vertices and similarity scores as edges, for sentence rank calculation

-> Finally, a certain number of top-ranked sentences form the final summary

## **Implementation**

### **Import Required Libraries**
First, import the libraries we’ll be leveraging for this challenge.

```
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
```

### **Read the Data**

Now let’s read our dataset. 

```
df = pd.read_csv("news.csv",encoding = 'unicode-escape')
df.head()
```

### **Split Text into Sentences**
Now the next step is to break the text into individual sentences. We will use the sent_tokenize( ) function of the nltk library to do this.
```
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)
```

### **Download GloVe Word Embeddings**

**GloVe** word embeddings are vector representation of words. These word embeddings will be used to create vectors for our sentences. We could have also used the Bag-of-Words or TF-IDF approaches to create features for our sentences, but these methods ignore the order of the words (and the number of features is usually pretty large).

We will be using the pre-trained Wikipedia 2014 + Gigaword 5 GloVe vectors available here. Heads up – the size of these word embeddings is 822 MB.

```
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
```
Let’s extract the words embeddings or word vectors.

```
# Extract word vectors
word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()
```

### **Text Preprocessing**

It is always a good practice to make your textual data noise-free as much as possible. So, let’s do some basic text cleaning.

```
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]
```
Get rid of the stopwords (commonly used words of a language – is, am, the, of, in, etc.) present in the sentences. If you have not downloaded nltk-stopwords, then execute the following line of code:

```
nltk.download('stopwords')
```
Now we can import the stopwords.

```
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
```
Let’s define a function to remove these stopwords from our dataset.

```
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new
```
```
# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
```
We will use clean_sentences to create vectors for sentences in our data with the help of the GloVe word vectors.

### **Vector Representation of Sentences**

Now, let’s create vectors for our sentences. We will first fetch vectors (each of size 100 elements) for the constituent words in a sentence and then take mean/average of those vectors to arrive at a consolidated vector for the sentence.

```
# Making sentences vectors
  sentence_vectors = []
  for i in clean_sentences:
    if len(i) != 0:
      v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)
```

### **Similarity Matrix Preparation**

The next step is to find similarities between the sentences, and we will use the cosine similarity approach for this challenge. Let’s create an empty similarity matrix for this task and populate it with cosine similarities of the sentences.

Let’s first define a zero matrix of dimensions (n * n).  We will initialize this matrix with cosine similarity scores of the sentences. Here, n is the number of sentences.

```
# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])
```
We will use Cosine Similarity to compute the similarity between a pair of sentences.

```
for i in range(len(sentences)):
  for j in range(len(sentences)):

    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
```

### **Applying PageRank Algorithm**
Before proceeding further, let’s convert the similarity matrix sim_mat into a graph. The nodes of this graph will represent the sentences and the edges will represent the similarity scores between the sentences. On this graph, we will apply the PageRank algorithm to arrive at the sentence rankings.

```
import networkx as nx

#Ranking lines using PageRank Algorithm
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
```

### **Summary Extraction**
Now we'll extract the top ranked lines from the original text in the given percentage of text.

```
 # Specify number of sentences to form the summary
  sn = int(len(sentences)*(summary_text_percent))
  
  # Generate summary
  summary_text = ''
  for i in range(sn):
    summary_text+=ranked_sentences[i][1]
  removed_lines=''
  for i in range(sn,len(ranked_sentences)):
    removed_lines+=ranked_sentences[i][1]
```

## **All at one place**

```
def Summary(text,summary_text_percent):
  sentences = sent_tokenize(text)
  # remove punctuations, numbers and special characters
  clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

  # make alphabets lowercase
  clean_sentences = [s.lower() for s in clean_sentences]

  stop_words = stopwords.words('english')

  # remove stopwords from the sentences
  clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

  #Making sentences vectors
  sentence_vectors = []
  for i in clean_sentences:
    if len(i) != 0:
      v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
    else:
      v = np.zeros((100,))
    sentence_vectors.append(v)

  # similarity matrix
  sim_mat = np.zeros([len(sentences), len(sentences)])
  for i in range(len(sentences)):
    for j in range(len(sentences)):
      if i != j:
        sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
  
  
  #Ranking lines using PageRank Algorithm
  nx_graph = nx.from_numpy_array(sim_mat)
  scores = nx.pagerank(nx_graph)
  ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
  
  # Specify number of sentences to form the summary
  sn = int(len(sentences)*(summary_text_percent))
  
  # Generate summary
  summary_text = ''
  for i in range(sn):
    summary_text+=ranked_sentences[i][1]
  removed_lines=''
  for i in range(sn,len(ranked_sentences)):
    removed_lines+=ranked_sentences[i][1]

  return [summary_text,removed_lines]
```

## **Test Drive?**

### **Original Text:**

```
RM aka Kim Namjoon was the first member to join BTS. The group released their debut single album 2 Cool 4 Skool on June 12, 2013. Apart from RM, BTS also features Jin, Suga, J-Hope, Jimin, V, and Jungkook. In an interview with Hypebeast, RM said, "This is the most difficult question to answer truthfully. Recently, I watched Everything Everywhere All At Once. That film visualized many of the ideas that Iâve had, such as the idea about multiple versions of myself existing based on small choices I made. I often think about what it would have been like if I continued my studies or became something other than a musician." "To be honest, one decision that I had often thought about was my choice to become a part of a boy band. In the late 2000s, musicians like Zico, Changmo, and Giriboy were the people that I started out with. In my journey with BTS, I drifted further and further away from that world and was tormented by the thought that the people that I liked â and the people who enjoyed the same music as I â did not have any love for me. I often wondered whether I made the right decision by joining a boy band. At the time, BTS, was treated like a complete outsider in the Korean hip-hop community. That stressed me out. I was constantly thinking about how I would be able to overcome that perception and how to define music or hip-hop,â he added. RM released his first solo mixtape in 2015. Three years later, he released his second mixtape, Mono. RM has collaborated with artists such as Wale, Younha, Warren G, Gaeko, Krizz Kaliko, MFBTY, Fall Out Boy, Primary, Lil Nas X, Erykah Badu, and Anderson .Paak. Earlier this month, RM released his first full-length solo album Indigo.
```

### **Summary Text:**

```
"To be honest, one decision that I had often thought about was my choice to become a part of a boy band.I often wondered whether I made the right decision by joining a boy band.I often think about what it would have been like if I continued my studies or became something other than a musician."At the time, BTS, was treated like a complete outsider in the Korean hip-hop community.I was constantly thinking about how I would be able to overcome that perception and how to define music or hip-hop,â he added.Earlier this month, RM released his first full-length solo album Indigo.In my journey with BTS, I drifted further and further away from that world and was tormented by the thought that the people that I liked â and the people who enjoyed the same music as I â did not have any love for me.In the late 2000s, musicians like Zico, Changmo, and Giriboy were the people that I started out with.Three years later, he released his second mixtape, Mono.That film visualized many of the ideas that Iâve had, such as the idea about multiple versions of myself existing based on small choices I made.
```

