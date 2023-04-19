#pip install requests beautifulsoup4 transformers sentencepiece #install the libraries needed

#import the libraries needed
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 1: Web Scraping
url = "https://en.wikipedia.org/wiki/Shinee" #the wikipedia page we are trying to scrap
html_page = requests.get(url).content
soup = BeautifulSoup(html_page)
text = ' '.join(map(lambda p: p.text, soup.find_all('p'))) # extract the text content of all paragraphs in the webpage

# Step 2: Text Preprocessing
#fn to clean text
def clean_text(text):
    # Remove reference links
    text = re.sub(r'\[.*?\]+', '', text)
    # Remove non-breaking spaces
    text = re.sub(r'\s+', ' ', text)
    return text

text = clean_text(text)

# Step 3: Chunking and Question-Answer Pair Generation
#fn to generate qa pairs
def generate_qa_pairs(text, chunk_size=3000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] # divide the cleaned text into chunks of given size
    qa_pairs = [] # initialize a list to store generated question-answer pairs
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large", max_length=1024)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    for chunk in chunks:
        input_ids = tokenizer.encode("generate questions about: " + chunk, return_tensors="pt", max_length=1024, truncation=True) # encode the chunk of text with a prompt to generate questions
        outputs = model.generate(input_ids=input_ids, max_length=1024, num_beams=20, early_stopping=True, num_return_sequences=5) 
        output_str = tokenizer.decode(outputs[0], skip_special_tokens=True) # decode the generated questions from tokenized format to string format
        questions = output_str.split("<sep>")
        for question in questions:
            input_text = "answer: " + chunk + " context: " + question
            input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True) # encode the input text with prompt "answer:"
            output = model.generate(input_ids=input_ids, max_length=1024, num_beams=20, early_stopping=True)
            answer = tokenizer.decode(output[0], skip_special_tokens=True) # decode the generated answer from tokenized format to string format
            qa_pairs.append({"question": question.strip(), "answer": answer.strip()}) # store the generated question-answer pair in the list
            print("Question: ", question.strip())
            print("Answer: ", answer.strip())
    return qa_pairs

#generate qa pairs from our wiki page
qa_pairs = generate_qa_pairs(text, chunk_size=3000)

# Step 4: Output
#fn to save pairs in csv format
def save_qa_pairs(qa_pairs):
    df = pd.DataFrame(qa_pairs)
    df.to_csv("shinee_qa_pairs.csv", index=False)
    
#save output as csv
save_qa_pairs(qa_pairs)