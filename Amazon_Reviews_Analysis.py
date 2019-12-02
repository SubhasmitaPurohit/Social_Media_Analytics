#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import nltk

import re
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer

import spacy

import gensim
from gensim import corpora


# In[3]:


get_ipython().run_line_magic('cd', 'F:\\BA\\Term 5\\Social Media Analysis')


# In[4]:


df=pd.read_csv("scrappingfinal.csv",sep=',')


# In[5]:


df


# In[6]:


df1=df


# In[7]:


df1.shape


# In[8]:


df1=df1.drop_duplicates()


# In[9]:


df1.shape


# In[10]:


df1['reviews'].value_counts()


# In[11]:


type(df1['reviews'])


# In[12]:


def getlength(reviews):
    reviews_tokens=reviews.split(" ")
    return len(reviews_tokens)


# In[13]:


df1['reviews_len']=df1['reviews'].apply(lambda x:getlength(x))
df1['reviews_len']


# In[14]:


df1


# In[15]:


df2=df1


# In[16]:


df2['reviews_len'].describe()


# In[17]:


STOPWORDS=stopwords.words("english")


# In[18]:


text_cleaned=[]
def clean_text(text):
    ps=PorterStemmer()
    text1=re.sub("http\S+","",text)
    text2=re.sub("@[\w]*","",text1)
    text3=text2.lower()
    #text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    #remove extra white space
    
    text_cleaned="".join([x for x in text3 if x not in string.punctuation])
    
    text_cleaned=re.sub(' +', ' ', text_cleaned)
    text_cleaned=text_cleaned.lower()
    tokens=text_cleaned.split(" ")
    tokens=[token for token in tokens if token not in STOPWORDS]
    text_cleaned=" ".join([ps.stem(token) for token in tokens])
    
    
    return text_cleaned


# In[19]:


df2['cleaned_reviews']=df2['reviews'].apply(lambda x:clean_text(x))
df2.head()


# In[20]:


def getlength(cleaned_reviews):
    cleaned_reviews_tokens=cleaned_reviews.split(" ")
    return len(cleaned_reviews_tokens)


# In[21]:


df2['cleaned_reviews_len']=df2['cleaned_reviews'].apply(lambda x:getlength(x))
df2['cleaned_reviews_len']


# In[22]:


df2


# In[38]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=1200,
        max_font_size=50, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=50)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(df2['cleaned_reviews'])


# In[23]:


corpus=df2['cleaned_reviews'].tolist()


# In[24]:


corpus


# In[25]:


tokenised_corpus=[]
for corp in corpus:
    
    tokenised_corpus.append([token for token in corp.split(" ")])
tokenised_corpus


# In[26]:


# Options for pandas
pd.options.display.max_columns = None
pd.options.display.max_rows = None

pd.options.display.max_colwidth=-1

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[27]:


dictionary = corpora.Dictionary(tokenised_corpus)


# In[28]:


len(dictionary)


# In[29]:


dictionary[0]


# In[30]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenised_corpus]


# In[31]:


doc_term_matrix[1]


# In[32]:


dictionary[15]


# In[39]:


LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=4, random_state=100,
                chunksize=1000, passes=50)


# In[40]:


lda_model.print_topics()


# In[42]:


import pyLDAvis
import pyLDAvis.gensim


# In[43]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis


# In[44]:


lda_model[doc_term_matrix[1]] #Gets the topic distribution of first document


# In[ ]:


df3=df2


# In[45]:


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df3 = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df3 = sent_topics_df3.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df3.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df3 = pd.concat([sent_topics_df3, contents], axis=1)
    return(sent_topics_df3)


df3_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=corpus)

# Format
df3_dominant_topic = df3_topic_sents_keywords.reset_index()
df3_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df3_dominant_topic.head(10)


# In[46]:


lda_model.log_perplexity(doc_term_matrix) #Perplexoity, lower the better


# In[47]:


from gensim.models import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenised_corpus, dictionary=dictionary,coherence="c_v")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[48]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics, random_state=100,chunksize=1000, passes=50)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=tokenised_corpus, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[49]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=tokenised_corpus, start=2, limit=10, step=2)


# In[50]:


limit=10
start=2
step=2
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[51]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", cv)


# In[ ]:




