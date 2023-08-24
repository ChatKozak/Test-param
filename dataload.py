#!/usr/bin/env python
# coding: utf-8

# # Document Loading

# ## Note to students.
# During periods of high load you may find the notebook unresponsive. It may appear to execute a cell, update the completion number in brackets [#] at the left of the cell but you may find the cell has not executed. This is particularly obvious on print statements when there is no output. If this happens, restart the kernel using the command under the Kernel tab.

# ## Retrieval augmented generation
#  
# In retrieval augmented generation (RAG), an LLM retrieves contextual documents from an external dataset as part of its execution. 
# 
# This is useful if we want to ask question about specific documents (e.g., our PDFs, a set of videos, etc). 

# ![overview.jpeg](attachment:overview.jpeg)

# In[3]:


get_ipython().system('pip install python-dotenv')


# In[13]:


get_ipython().system('pip install pypdf')


# In[10]:


get_ipython().system(' pip install langchain')


# In[7]:


import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = "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"


# ## PDFs
# 
# Let's load a PDF [transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf) from Andrew Ng's famous CS229 course! These documents are the result of automated transcription so words and sentences are sometimes split unexpectedly.

# In[8]:


# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
#! pip install pypdf 


# In[15]:


from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("devops_ref-2.pdf")
pages = loader.load()


# Each page is a `Document`.
# 
# A `Document` contains text (`page_content`) and `metadata`.

# In[16]:


len(pages)


# In[17]:


page = pages[0]


# In[18]:


print(page.page_content[0:500])


# In[19]:


page.metadata

