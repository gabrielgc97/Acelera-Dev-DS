#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[3]:


# Algumas configurações para o matplotlib.
# %matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


countries = pd.read_csv("countries of the world.csv")


# In[5]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# In[114]:


countries.info


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[6]:


num_cols=['Population','Area','Pop_density','Coastline_ratio','Net_migration',"Infant_mortality",'GDP','Literacy','Phones_per_1000','Arable','Crops','Other','Climate','Birthrate','Deathrate','Agriculture','Industry','Service']

for C in num_cols:
    countries[C]=countries[C].apply(lambda x: str(x).replace(',','.'))
    countries[C]=countries[C].astype(np.float)


# In[7]:


countries.Country=countries.Country.apply(lambda x: str(x).strip())
countries.Region=countries.Region.apply(lambda x: str(x).strip())


# In[8]:


countries.shape


# In[51]:


countries.isna().sum()


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[6]:


def q1():
    # Retorne aqui o resultado da questão 1.
    a=list(countries.Region.unique())
    a.sort()
    return a


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[93]:


from sklearn.preprocessing import (
    OneHotEncoder, Binarizer, KBinsDiscretizer,
    MinMaxScaler, StandardScaler, PolynomialFeatures
)


# In[94]:


discretizer = KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
discretizer.fit(countries[["Pop_density"]])


# In[95]:


discretizer.bin_edges_[0]


# In[102]:


def q2():
    return countries["Pop_density"][countries.Pop_density>=discretizer.bin_edges_[0][9]].count()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[101]:


def q3():
    reg_at=countries.Region.nunique()
    Climate_at=countries.Climate.nunique()
    return reg_at+Climate_at


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[127]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[139]:


df_pad=countries[["Country","Region"]]


# In[128]:


num_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy='median')),
    ("stardizer", StandardScaler())
])


# In[129]:


countries_pipetransf= num_pipeline.fit_transform(countries[num_cols])

countries_pipetransf[:10]


# In[140]:


countries_pad=pd.DataFrame(countries_pipetransf,columns=num_cols)
df_pad=pd.concat([df_pad,countries_pad],axis=1)


# In[141]:


df_pad


# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[142]:


def q4():
    return test_country[11]


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[9]:


quant1=countries.Net_migration.quantile(0.25)
quant3=countries.Net_migration.quantile(0.75)
IQR=quant3-quant1


# In[14]:


outliers_abaixo=countries['Net_migration'][countries.Net_migration<quant1-1.5*IQR].count()
outliers_acima=countries['Net_migration'][countries.Net_migration>quant3+1.5*IQR].count()


# In[16]:


outliers_abaixo


# In[161]:


def q5():
    return tuple([outliers_abaixo,outliers_acima,False])


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[18]:


from sklearn.datasets import fetch_20newsgroups


# In[19]:


from sklearn.feature_extraction.text import (
    CountVectorizer, TfidfTransformer, TfidfVectorizer
)


# In[20]:


categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[21]:


cv=CountVectorizer()
newgroups_counts=cv.fit_transform(newsgroup.data)


# In[22]:


freqs=pd.DataFrame(newgroups_counts.toarray(),columns=np.array(cv.get_feature_names()))


# In[23]:


def q6():
    return freqs['phone'].sum()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[24]:


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(newsgroup.data)
newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroup.data)


# In[25]:


newsgroups_tfidf=pd.DataFrame(newsgroups_tfidf_vectorized.toarray(),columns=np.array(cv.get_feature_names()))


# In[208]:


def q7():
    return float(round(newsgroups_tfidf['phone'].sum(),3))

