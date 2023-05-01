# TestingDeltaCognition
## Introduction
This is a testing task of Delta Cognition about Question Answering system for research Paper.
Home challenge for the technical part.
 
---START---
The purpose of this task is to implement the machine learning models that provide the descriptive answer to the given query based on the list of research papers. For example, the given query can be “What is Graph Neural Network?”, and the corresponding answer will be the description that is similar to this “Graph Neural Network (GNN) is a class of ANN for processing data that can be presented as graphs. The latest research work related to GNN includes [1] … [2] … [3] …”. 
 
Input: A query 
Output: The descriptive answer to the given query based on the list of research papers. 
 
Required tech stack: 
Framework: PyTorch or Tensorflow 
API: FastAPI (Python) 
Presentation: Gradio (Python) 
Database: SQL or NoSQL 
---END---
## Quick Start

This code provides a Python implementation of a question-answering model that uses the ArXiv API to retrieve research papers relevant to the given query and then preprocesses the text of the papers for answering questions using the [Hugging Face Transformers](https://huggingface.co) library. The code uses the [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) library to parse the XML returned by the [ArXiv](https://arxiv.org) API, the [NLTK](https://nltk.org) library for natural language processing, and the [spaCy](https://spacy.io) library for part-of-speech tagging.

**Build QA system**

* Clone Code
  ```bash
  git clone https://github.com/hoangpnhat/TestingDeltaCognition.git
  cd QAsystem
  ```
* Create a virtual environment (optional)
  ```bash
  virtualenv QAsystem
  cd /path/to/venv/QAsystem
  source ./bin/activate
  ```
* Install other requirements. 
  ```python
  python3 -m pip install -r requirements.txt
  ```
**Usage**

Run the following command to launch the Gradio interface:
```python
gradio main.py
```
