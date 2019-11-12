#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup


# In[2]:


header={'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.70 Safari/537.36'}


# In[3]:


cookie={}
def getAmazonSearch(search_query):
    url="https://www.amazon.in/s?k="+search_query
    print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"
    #print(page.content)


# In[4]:


product=[]
response=getAmazonSearch('xiaomi+mobile+phones')
#beautifulsoup is a package
soup=BeautifulSoup(response.content)
for i in soup.findAll("span",{'class':'a-size-medium a-color-base a-text-normal'}):
    product.append(i.text)


# In[5]:


product


# In[6]:


def Search(search_query):
    url="https://www.amazon.in/dp/"+search_query
    print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"


# In[7]:


data_asin=[]
response=getAmazonSearch('xiaomi+mobile+phones')
#beautifulsoup is a package
soup=BeautifulSoup(response.content)
for i in soup.findAll("div",{'class':"sg-col-20-of-24 s-result-item sg-col-0-of-12 sg-col-28-of-32 sg-col-16-of-20 sg-col sg-col-32-of-36 sg-col-12-of-16 sg-col-24-of-28"}):
    data_asin.append(i['data-asin'])


# In[8]:


data_asin


# In[ ]:


len(data_asin)


# In[ ]:


def Search_1(search_query):
    url="https://www.amazon.in"+search_query
    print(url)
    page=requests.get(url,cookies=cookie,headers=header)
    if page.status_code==200:
        return page
    else:
        return "Error"


# In[ ]:


link=[]
for i in range(len(data_asin)-1):
    response=Search(data_asin[i+1])
#beautifulsoup is a package
    soup=BeautifulSoup(response.content)
    for i in soup.findAll("a",{'data-hook':"see-all-reviews-link-foot"}):
        link.append(i['href'])


# In[ ]:


len(link)


# In[ ]:


r1=[]
for j in range(len(link)):
    for k in range(100):
        response=Search_1(link[j]+'&pageNumber='+str(k))
#beautifulsoup is a package
        soup=BeautifulSoup(response.content)
        for i in soup.findAll("span",{'data-hook':"review-body"}):
            r1.append(i.text)


# In[ ]:


len(r1)


# In[ ]:


r1


# In[ ]:


import pandas as pd


# In[ ]:


r={'reviews':r1}


# In[ ]:


df=pd.DataFrame.from_dict(r)
df


# In[ ]:


df.to_csv('5000scrapping.csv',index=False)


# In[ ]:


len(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




