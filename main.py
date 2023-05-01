import requests
from bs4 import BeautifulSoup
import re
import nltk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering,pipeline
import torch
import arxiv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gradio as gr
import spacy

# Use GPU if available
device = torch.device("cuda")
nltk.download('punkt')
# Crawl researcher papers
def extract_keyword(query):
    # Load the language model
    nlp = spacy.load("en_core_web_sm")

    # Process the question
    doc = nlp(query)
    # Extract the nouns and proper nouns as keywords
    keywords = []
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            keywords.append(token.text)
    return keywords
def get_papers(keywords, num_papers):
    query=""
    for k in keywords:
        query+= k +" "
    # Format the query for use in the ArXiv API
    query = query.replace(' ', '+')

    # Build the URL for the ArXiv API with the query and number of papers
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_papers}&order=submitted_date"
    headers = requests.utils.default_headers()
    headers.update(
    {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    }
)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "xml")
    summary = soup.find_all("summary")
    abstract =[]
    for summa in summary:
        abstract.append(summa.text)
    return abstract

# Define a function to preprocess the papers for question answering tasks
def preprocess_papers(papers_abstract):
    # Define a list to store the preprocessed papers
    preprocessed_papers = []
    
    # Loop through each paper and preprocess the text
    for paper in papers_abstract:
        # Tokenize the text into sentences and words

        sentences = sent_tokenize(paper)

        preprocessed_papers.append(sentences)
    
    return preprocessed_papers
# Example usage

# Preprocess the papers
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(device)



def answer_question(question, sentences):

    # Find the best sentence
    best_sentence = None
    best_score = float('-inf')
    for i, sentence in enumerate(sentences):
        inputs = tokenizer.encode_plus(question, sentences[i], add_special_tokens=True, return_tensors='pt').to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        start_scores, end_scores = model(input_ids, attention_mask=attention_mask).values()
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        score = start_scores[0][answer_start].item() + end_scores[0][answer_end-1].item()

        if score > best_score:
            best_score = score
            best_sentence = sentences[i]

    return best_sentence



def question_answer(query):
    num_papers = 5
    keywords = extract_keyword(query)

    papers_abstract = get_papers(keywords, num_papers)
    processed_papers = preprocess_papers(papers_abstract)
    anwers = []
    for paper in processed_papers:
        answer = answer_question(query, paper)
        print("Question:", query)
        print("Answer:", answer)
        print("-" * 50)
        anwers.append(answer)
    result = ""
    for r in anwers:
        result += r +" "
    return result


#     # pass  # Implement your question-answering model here...


demo = gr.Interface(fn=question_answer, 
            inputs=gr.Textbox(lines=2, placeholder="Query..."),

            outputs=["textbox"]).launch()

demo.launch() 