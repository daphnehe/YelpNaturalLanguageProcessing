# -*- coding: utf-8 -*-

import pandas as pd
import string

# Create sample DataFrame
dataframe = [
    (0, "Spark is great!!!"),
    (1, "We are learning Spark!"),
    (2, "Spark is better than hadoop,  no doubt.")
]

df=pd.DataFrame(dataframe, columns=['id', 'sentence'])
df

df.loc[:, 'sentence'].str.split(' ')

# Lowercase
sentences=df.loc[:, 'sentence']
tokenized=[]
for i, each_sentence in sentences.iteritems(): 
  word_list=each_sentence.lower().split(' ')
  tokenized.append(word_list)
tokenized
df.loc[:, 'tokenized']=tokenized
df

# Remove punctuations and numbers
# import string

# punct="!,.[]()<>@"
# sentences=df.loc[:, 'sentence']
# tokenized=[]
# for i, each_sentence in sentences.iteritems(): 
#   for character in each_sentence: 
#     if not character.isalpha(): 
#       each_sentence=each_sentence.replace(character, '')
  # word_list=each_sentence.lower().split(' ')
#   tokenized.append(word_list)
# tokenized
# df.loc[:, 'tokenized']=tokenized
# df



punct="!,.[]()<>@"
sentences=df.loc[:, 'sentence']
tokenized=[]
for i, each_sentence in sentences.iteritems(): 
  for character in each_sentence: 
    if character in punct: 
      each_sentence=each_sentence.replace(character, '')
  word_list=each_sentence.lower().split(' ')
  tokenized.append(word_list)
tokenized
df.loc[:, 'tokenized']=tokenized
df

# Remove stop words

stop_words_list=['an', 'no', 'the', 'is', 'are']
no_stop_words=[]

for i, each_tokenized in df.loc[:, 'tokenized'].iteritems(): 
  tokenized=[]
  for each_word in each_tokenized: 
    if not each_word in stop_words_list: 
      tokenized.append(each_word)
  no_stop_words.append(tokenized)
df.loc[:, 'no_stop_words']=no_stop_words
df

# Remove stop words
from sklearn.feature_extraction import text
stop_words=text.ENGLISH_STOP_WORDS
no_stop_words=[]

for i, each_tokenized in df.loc[:, 'tokenized'].iteritems(): 
  tokenized=[]
  for each_word in each_tokenized: 
    if not each_word in stop_words: 
      tokenized.append(each_word)
  no_stop_words.append(tokenized)
df.loc[:, 'no_stop_words']=no_stop_words
df

# Split words
series_list=[]
for i, each_tokenized in df.loc[:, 'no_stop_words'].iteritems(): 
  sample=each_tokenized
  count_dict={}
  for each_word in sample: 
    if each_word in count_dict:
      count_dict[each_word]=count_dict[each_word]+1
    else: 
      count_dict[each_word]=1
  srs=pd.Series(count_dict)
  series_list.append(srs)
series_list

# Join words into a training dataset
pd.concat(series_list, axis=1).T

import wordcloud
import matplotlib.pyplot as plt
test_string='Transcription is the first of several steps of DNA based gene expression in which a particular segment of DNA is copied into RNA by the enzyme RNA polymerase. Both DNA and RNA are nucleic acids, which use base pairs of nucleotides as a complementary language'
# my_wordcloud=wordcloud.WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(df['sentence'][0])
my_wordcloud=wordcloud.WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(test_string)
# wordcloud=wordcloud.WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(sample))
# wordcloud=wordcloud.WordCloud(width=800, height=800, background_color='white', min_font_size=20).generate_from_frequencies(top_terms_dict)   
plt.imshow(my_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Using Naive Bayes Model to Predict Review
import pandas as pd

df=pd.read_csv('/content/yelp_reviews.csv')
print(df.shape)
df.head()

from sklearn.feature_extraction.text import CountVectorizer

# a new class of a model/vectorizer
# model/vectorizer.fit
# model.predict or vectorizer.transform
cv=CountVectorizer()

text_data=df.loc[:, 'text']
cv.fit_transform(text_data)

word_list=cv.get_feature_names()
word_list

cv.vocabulary_

plt.figure(figsize=(20, 5))
pd.Series(cv.vocabulary_).sort_values(ascending=False).head(100).plot(kind='bar')

X = cv.fit_transform(text_data).toarray()
input_df=pd.DataFrame(X, columns=word_list)
plt.figure(figsize=(20, 5))
input_df.sum().sort_values(ascending=False).head(100).plot(kind='bar')

# Add a word counter feature

from sklearn.naive_bayes import GaussianNB

# new model object
# model.fit
# model.train

nb=GaussianNB()

y=df.loc[:, 'class']

nb.fit(X, y)

sample_text=df.iloc[50]['text']

nb.predict(X[50])

sample_array=sample_text.split(' ')

sample_input=cv.transform(sample_array)

nb.predict(sample_input.toarray())
