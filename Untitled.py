#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[44]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[19]:


data=datasets.load_breast_cancer()


# In[22]:


x,y=data.data,data.target


# In[26]:


n_sem,n_feg=x.shape


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[28]:


sc=StandardScaler()


# In[29]:


x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[32]:


x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))


# In[35]:


y_test.shape


# In[36]:


y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)


# In[37]:


y_train.shape


# In[38]:


class LogiReg(nn.Module):
    def __init__(self,n_feg):
        super(LogiReg,self).__init__()
        self.lin=nn.Linear(n_feg,1)
    def forward(self,x):
        y_pre=torch.sigmoid(self.lin(x))
        return y_pre
model=LogiReg(n_feg)


# In[40]:


learing_rate=0.01
loss=nn.BCELoss()
opt=torch.optim.SGD(model.parameters(),lr=learing_rate)


# In[43]:


num_epoch=100
for i in range(num_epoch):
    y_pre=model(x_train)
    ls=loss(y_pre,y_train)
    ls.backward()
    opt.step()
    opt.zero_grad()
    if (i+1)%10==0:
        print(f"epoch:{i+1}  loss:{ls.item():.4f}")
with torch.no_grad():
    y_pre_test=model(x_test)
    y_pre_cls=y_pre_test.round()
    acc=y_pre_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f"acc:{acc:.4f}")


# In[ ]:




