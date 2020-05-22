#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.shape


# In[7]:


df.info


# In[8]:


info = pd.DataFrame({'colunas': df.columns, 'dtypes' : df.dtypes, 'missing' : df.isna().sum() ,'missing%' : df.isna().sum()/df.shape[0]})


# In[9]:


info


# In[10]:


df.nunique()


# In[11]:


df.groupby('Age')['Gender'].value_counts()


# In[12]:


df_F= df[df['Gender']=='F']
df_F_26_35=df_F[df_F['Age']=='26-35']

df_F_26_35.shape[0]


# In[13]:


df.isna().max(axis=1).sum()/df.shape[0]


# In[14]:


df.isna().sum().max()


# In[15]:


df['Product_Category_3'][df['Product_Category_3'].notna()].mode()[0]


# ## PADRONIZAÇÃO E NORMALIZAÇÃO PURCHASE

# In[16]:


mean = df.Purchase.mean()
std = df.Purchase.std()


# In[17]:


df['Purchase_P'] = (df['Purchase'] - mean)/std


# In[19]:


df['Purchase_N'] = (df.Purchase - df.Purchase.min())/(df.Purchase.max()-df.Purchase.min())


# In[21]:


a8=df.Purchase_N.mean()


# In[18]:


a9=df['Purchase_P'][df['Purchase_P']<=1][df['Purchase_P']>=-1].shape[0]


# In[19]:


df[df['Product_Category_2'].isna()].shape[0]


# In[20]:


df[df['Product_Category_2'].isna()][df['Product_Category_3'].isna()].shape[0]


# In[21]:


df[df['Product_Category_2'].isna()].shape[0] == df[df['Product_Category_2'].isna()][df['Product_Category_3'].isna()].shape[0]


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[22]:


def q1():
    # Retorne aqui o resultado da questão 1.
    a1 = (537577, 12)
    return a1


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[23]:


def q2():
    # Retorne aqui o resultado da questão 2.
    a2 = df_F_26_35.shape[0]
    return a2


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[24]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return df['User_ID'].nunique()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[25]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return info['dtypes'].nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[26]:


def q5():
    # Retorne aqui o resultado da questão 5.
    return 0.694410


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[27]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return 373299


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[28]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return df['Product_Category_3'][df['Product_Category_3'].notna()].mode()[0]


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[29]:


def q8():
    # Retorne aqui o resultado da questão 8.
    return a8


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[30]:


def q9():
    # Retorne aqui o resultado da questão 9.
    return a9


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[31]:


def q10():
    # Retorne aqui o resultado da questão 10.
    return df[df['Product_Category_2'].isna()].shape[0] == df[df['Product_Category_2'].isna()][df['Product_Category_3'].isna()].shape[0]

