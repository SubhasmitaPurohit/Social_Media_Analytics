#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Manipulation
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


import nltk#Natural Language Tool Kit

import re
import string
from nltk.corpus import stopwords
from nltk import PorterStemmer

#!pip install spacy
import spacy

#Visualisation
import gensim
from gensim import corpora


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[3]:


get_ipython().run_line_magic('cd', 'C:\\Users\\Subhasmita Purohit\\Documents\\TweetScraper\\TweetScraper-master')


# In[4]:


#Read the csv file
df=pd.read_csv("EconomicSlowdown.csvTweets.csv", sep=',')


# In[5]:


#Display
df


# In[6]:


#to check no. of tweets
df.shape


# In[7]:


df.info()


# In[8]:


df['usernameTweet'].value_counts()


# In[9]:


df['datetime'].value_counts()


# ## Cleaning the tweets

# In[10]:


#assign to other variable so if lose the data in futeure we ca access the original data in df
df1=df


# In[11]:


df1.columns


# Here drop unnecessary columns like user id, medias, has_medias, datetime(this is fix Nov29 to Dec 2), is_reply, is_retweet,nbr_reply. Leave usernameaTweet, nbr_retweet and and nbr_favorite. Why?To explore then remove.

# In[12]:


#If feel like useful check before drop the columns
df1['is_reply'].value_counts()


# In[13]:


df1['is_retweet'].value_counts()
#Useless as all the records have the same values


# In[14]:


df1['nbr_retweet'].value_counts()


# In[15]:


df1['nbr_retweet'].describe()


# In[16]:


df1['nbr_favorite'].value_counts()


# In[17]:


df1['nbr_favorite'].describe()


# In[18]:


df1['has_media']


# In[19]:


#to check how many null values are there
df1['has_media'].isnull().sum()


# In[20]:


df1['medias'].isnull().sum()


# In[21]:


df1['medias']


# In[22]:


df1=df1.drop(['medias','has_media','user_id','url','is_reply','is_retweet','ID','nbr_reply'],axis=1)


# In[23]:


#after drop unnecessary columns check the size of the data frame
df1.shape


# In[24]:


df1.columns


# In[25]:


df1=df1.drop(['usernameTweet','nbr_retweet', 'nbr_favorite'],axis=1)


# In[26]:


df1.shape


# In[27]:


df1['text']


# In[28]:


#Final table on which main task will perform
df2=df1


# In[29]:


df2


# In[30]:


#Change the column name from text to tweets
df2=df2[['text']]
df2.columns=['tweet']


# In[31]:


df2


# In[32]:


#cross check the tweets
df2['tweet'].value_counts()


# In[33]:


#Calculate the length of each tweet
def getlength(tweets):
    tweets_tokens=tweets.split(" ")
    return len(tweets_tokens)


# In[34]:


#Create an another column to store length of each tweet
df2['tweet_len']=df2['tweet'].apply(lambda x:getlength(x))
df2['tweet_len']


# In[35]:


df2


# In[36]:


df2['tweet_len'].describe()


# In[37]:


sns.kdeplot(df2['tweet_len']).set_title(" Distribution of Length of Tweet")


# In[38]:


#For other hastags
def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)   


# In[39]:


df2['hashtags']=df2.tweet.apply(find_hashtags)


# In[40]:


df2


# In[41]:


df2['hashtags'].describe()


# In[42]:


# take the rows from the hashtag columns where there are actually hashtags
hashtags_list_df = df2.loc[
                       df2.hashtags.apply(
                           lambda hashtags_list: hashtags_list !=[]
                       ),['hashtags']]


# In[43]:


hashtags_list_df


# In[44]:


#create dataframe where each use of hashtag gets its own row
flattened_hashtags_df = pd.DataFrame(
    [hashtag for hashtags_list in hashtags_list_df.hashtags
    for hashtag in hashtags_list],
    columns=['hashtags'])


# In[45]:


flattened_hashtags_df


# In[46]:


# number of unique hashtags
flattened_hashtags_df['hashtags'].unique().size


# In[47]:


# count of appearances of each hashtag
popular_hashtags = flattened_hashtags_df.groupby('hashtags').size()                                        .reset_index(name='counts')                                        .sort_values('counts', ascending=False)                                        .reset_index(drop=True)


# In[48]:


popular_hashtags


# In[49]:


# number of times each hashtag appears
counts = flattened_hashtags_df.groupby(['hashtags']).size()                              .reset_index(name='counts')                              .counts

# define bins for histogram                              
my_bins = np.arange(0,counts.max()+200, 200)-0.5

# plot histogram of tweet counts
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+200, 200)
plt.xlabel('hashtag number of appearances')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


# In[50]:


#Removeurl
def removeurl(tweet,replace_tweet=""):
    text="r'^https?:\/\/.*[\r\n]*'"
    return re.sub(text,replace_tweet,tweet)


# In[51]:


df2['cleaned_tweet']=df2['tweet'].apply(lambda x:removeurl(x))


# In[52]:


#Removeurl
def removeURL(text,replace_text=""):
    re_url=" http:\\S+" #\S+ matched all non-whitespace characters
    return re.sub(re_url,replace_text,text)


# In[53]:


df2['cleaned_tweet']=df2['tweet'].apply(lambda x:removeURL(x))


# Convert tweets to lower case

# In[54]:


df2['cleaned_tweet']=df2['tweet'].apply(lambda x:x.lower())
df2


# In[55]:


df2['cleaned_tweet'].value_counts()


# Remove hashtags

# In[56]:


def removeHashTag(tweet,replace_tweet=""):
    re_hashtag="#\S+"
    return re.sub(re_hashtag,replace_tweet,tweet)


# In[57]:


df2['cleaned_tweet']=df2['cleaned_tweet'].apply(lambda x:removeHashTag(x))


# In[58]:


df2


# strips of the whitespaces at the end of the tweets

# In[59]:


df2['cleaned_tweet']=df2['cleaned_tweet'].apply(lambda x:x.strip())


# In[60]:


# strips of the : at the end of the tweets
df2['cleaned_tweet']=df2['cleaned_tweet'].apply(lambda x:x.strip(":")) 


# In[61]:


STOPWORDS=stopwords.words("english")
STOPWORDS


# In[62]:


import string
import spacy


# In[63]:



def removePunctuations(tweet,replace_tweet=""):
    tweet=tweet.replace("&amp","")
    return tweet.translate(str.maketrans('', replace_tweet, string.punctuation)).strip()


# In[64]:


df2['cleaned_tweet']=df2['cleaned_tweet'].apply(lambda x:removePunctuations(x))


# In[65]:


#Since this is economicslowdown and gdp growth data, this word will occur very frequently and must be removed

STOPWORDS.append('economicslowdown')
STOPWORDS.append('gdpgrowth')

nlp = spacy.load('en_core_web_sm')

def cleanup_text(docs,allowed_tags=['NOUN', 'PROPN']):
    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    if len(allowed_tags)==0:
        tokens = [tok.lemma_ for tok in doc if tok.lemma_ != '-PRON-']
    else:
        tokens = [tok.lemma_ for tok in doc if tok.pos_ in allowed_tags and tok.lemma_ != '-PRON-']
    tokens = [tok.lower().strip() for tok in tokens if tok.lower() not in STOPWORDS]
    
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return texts[0]


# In[66]:


df2['cleaned_text']=df2['cleaned_tweet'].apply(lambda x:cleanup_text(x,allowed_tags=[]))


# In[67]:


df2


# In[68]:


#Compare the length of raw and clean text 
def getlength(cleaned_text):
    cleaned_text_tokens=cleaned_text.split(" ")
    return len(cleaned_text_tokens)


# In[69]:


#Create an another column to store length of each cleaned_text
df2['cleaned_text_len']=df2['cleaned_text'].apply(lambda x:getlength(x))
df2['cleaned_text_len']


# In[70]:


df2


# Building WordCloud

# In[71]:


df2=df2.drop(['cleaned_tweet'],axis=1)


# In[72]:


test=""
def clean_text(text):
    ps=PorterStemmer()
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    #remove extra white space

    text_cleaned="".join([x for x in text if x not in string.punctuation])
    text_cleaned=re.sub('\n','',text_cleaned)
    text_cleaned=re.sub('\d+','',text_cleaned)
    
    

    text_cleaned=re.sub(' +', ' ', text_cleaned)
    
    tokens=text_cleaned.split(" ")
    tokens=[token for token in tokens if token not in STOPWORDS]
    text_cleaned=" ".join([ps.stem(token) for token in tokens])    
    
    return text_cleaned


print(clean_text(test))


# In[73]:


df2['cleaned_text']=df2['cleaned_text'].apply(lambda x:clean_text(x))
df2.head()


# In[74]:


df2_text=" ".join(df2['cleaned_text'])


# In[75]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[76]:


wordcloud = WordCloud(max_font_size=70, max_words=100, background_color="white",collocations=False).generate(df2_text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# From the word cloud we can see that people are mostly talking about gdp growth and economic goes down talk about India, Govt. Bjp, narendra modi in some corner, Mostly they speak all negative as word like low, bad are appear in the word cloud we can see that.

# As unigram doesn't give an accurate output we can try for 

# In[77]:


from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from operator import itemgetter


# In[78]:


def getBigrams(df2_text):
    df2_tokens=nltk.word_tokenize(df2_text)
    
    tweet=nltk.Text(df2_tokens)
    
    tweet=[word for word in tweet if word!="." and len(word)>1 and word!="'s"]
    
    finder = BigramCollocationFinder.from_words(tweet)
    
    bigram_measures = BigramAssocMeasures()
    
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    
    scoredList = sorted(scored, key=itemgetter(1), reverse=True)
    
    return scoredList


# In[79]:


scoredList=getBigrams(df2_text)
scoredList


# In[80]:


word_dict={}
listLen=len(scoredList)
for i in range(listLen):
    word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]


# In[81]:


WC_height =700
WC_width =700
WC_max_words =250
 
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color="white")
 
wordCloud.generate_from_frequencies(word_dict)
 
plt.title("frequently occurring bigrams")
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Improving Bigram

# In[82]:


allowed_postags=['NOUN', 'PROPN', 'NUM']
df2['cleaned_text_2']=df2['cleaned_text'].apply(lambda x:cleanup_text(x,allowed_tags=allowed_postags))


# In[83]:


df2[['tweet','cleaned_text_2']].head()


# In[84]:


df2_text_2=" ".join(df2['cleaned_text_2'])


# In[85]:


scoredList_2=getBigrams(df2_text_2)
scoredList_2


# In[86]:


#After so much cleaning, mining and manipulationalso there are lots of unnecessary words are present which create problem at the end so we have to identifiin how many rows they are present
df2[df2['cleaned_text'].str.contains("httpswww","twittercom")]


# In[87]:


df2['cleaned_text']=df2['cleaned_text'].apply(lambda x:x.replace("httpswww",""))
df2['cleaned_text']=df2['cleaned_text'].apply(lambda x:x.replace("twittercom",""))
df2['cleaned_text']=df2['cleaned_text'].apply(lambda x:x.replace("comnewsbusi",""))
df2['cleaned_text_2']=df2['cleaned_text'].apply(lambda x:cleanup_text(x,allowed_tags=allowed_postags))


# In[88]:


df2_text_2=" ".join(df2['cleaned_text_2'])
scoredList_2=getBigrams(df2_text_2)
scoredList_2


# In[89]:


word_dict_2={}
listLen=len(scoredList_2)
for i in range(listLen):
    word_dict_2['_'.join(scoredList_2[i][0])] = scoredList_2[i][1]


# In[90]:



WC_height=700
WC_width=700
WC_max_words=250
 
wordCloud=WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width,background_color="white")
 
wordCloud.generate_from_frequencies(word_dict_2)
 
plt.title("frequently occurring bigrams")
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Check the previous biagram and the recent one we can identify the difference. Gdp_groeth is still in the word cloud with that many new words are coming which making more sense.

# ## Clustering

# In[91]:


from sklearn.feature_extraction.text import CountVectorizer


# In[92]:


df2['cleaned_text']=df2['cleaned_text'].apply(lambda x:cleanup_text(x,allowed_tags=[]))


# In[93]:


df3=df2.drop_duplicates(subset=['cleaned_text'])


# In[94]:


df3.shape


# In[95]:


text=df3['cleaned_text'].tolist()


# In[96]:


#create the transform
vectorizer = CountVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
#print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)

# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())


# In[97]:


from sklearn.cluster import KMeans


# In[98]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(vector)
    Sum_of_squared_distances.append(km.inertia_)


# In[99]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[100]:


kmeans = KMeans(n_clusters=7).fit(vector)


# In[101]:


df3['BoW_Clusters']=kmeans.labels_


# In[102]:


df3['BoW_Clusters'].value_counts()


# In[103]:


df3[df3['BoW_Clusters']==3].head(30)


# Clustering using TF-IDF

# In[104]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[105]:


# create the transform
tfidf_vectorizer = TfidfVectorizer()

# tokenize and build vocab
tfidf_vectorizer.fit(text)

# summarize
#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)
# encode document
tfidf_vector = tfidf_vectorizer.transform(text)

# summarize encoded vector
print(tfidf_vector.shape)
print(tfidf_vector.toarray())


# In[106]:


Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(tfidf_vector)
    Sum_of_squared_distances.append(km.inertia_)


# In[107]:


plt.plot(K, Sum_of_squared_distances,'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k using TF-IDF')
plt.show()


# In[108]:


kmeans_tfidf = KMeans(n_clusters=12).fit(tfidf_vector)


# In[109]:


df3['TfIdf_Clusters']=kmeans_tfidf.labels_


# In[110]:


df3['TfIdf_Clusters'].value_counts()


# In[111]:


df3[df3['TfIdf_Clusters']==3].head(20)


# cluster size 3 looks good moving ahead for sentiment analysis

# Sentiment Analysis

# In[112]:


#!pip install vaderSentiment


# In[113]:


#!pip install nltk


# In[114]:


#nltk.download()


# In[115]:


from nltk.sentiment import vader
from nltk.sentiment.util import *

from nltk import tokenize

sid = vader.SentimentIntensityAnalyzer()


# To understand what people talk about Economic slowdown and Gdo growth we are doing sentiment analysis. Basicallu it's about their opinion on GDP growth and economic slodown and express there expectation or dissatisfaction about the current Govt.

# In[118]:


df3['sentiment_compound_polarity']=df3['tweet'].apply(lambda x:sid.polarity_scores(x)['compound'])

df3['sentiment_negative']=df3['tweet'].apply(lambda x:sid.polarity_scores(x)['neg'])
df3['sentiment_pos']=df3['tweet'].apply(lambda x:sid.polarity_scores(x)['pos'])
df3['sentiment']=''
df3.loc[df3.sentiment_compound_polarity>=0,'sentiment']="positive"
df3.loc[df3.sentiment_compound_polarity==0,'sentiment']="neutral"

df3.loc[df3.sentiment_compound_polarity<0,'sentiment']="negative"


# In[119]:


df3.head()


# In[120]:


df3['sentiment'].value_counts()


# In[121]:


sns.countplot(df3['sentiment'])


# In[122]:


df3[df3['sentiment']=="negative"].head()


# ### Conclusion
# from the word cloud we concluded that people are mostly talking negative about the economic slowdown where as fron the clustering and Tf idf we conclude that people are postly talk positively about the Govt. At this time how Govt. trying hard to balnce the economic condistions by different policies. In most of the tweet they mention about Prime misister and the finance minister.

# In[ ]:




