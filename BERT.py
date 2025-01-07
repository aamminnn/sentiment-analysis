from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Encode and calculate sentiment
# tokens = tokenizer.encode('I hated this, absolutely worst', return_tensors='pt')
tokens = tokenizer.encode('I love this, absolutely great', return_tensors='pt')
print(tokens)
tokenizer.decode(tokens[0])

result = model(tokens)
print(result)
print(result.logits)
print(torch.argmax(result.logits))
print(int(torch.argmax(result.logits))+1)


# Collect reviews
link=r'https://www.google.com/search?gs_ssp=eJwNyUEOQDAQAMC4SrxBL852tQRP8Ittu6SplijS5zPXKat2axHfdL3DGBIUcwNZojFKA69WwQDcz5B5xAkUW5Qdyj-XOlJywvNF-hHh8JG0OCne5ISlncIHaGEbJA&q=nasi+kerabu+moknab+pantai+dalam&rlz=1C5GCCM_en&oq=Nasi+kerabu+mok&gs_lcrp=EgZjaHJvbWUqEAgBEC4YrwEYxwEYgAQYjgUyCggAEAAYsQMYgAQyEAgBEC4YrwEYxwEYgAQYjgUyCQgCEEUYORiABDITCAMQLhivARjHARixAxiABBiOBTIQCAQQLhivARjHARiABBiOBTIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIQCAgQLhivARjHARiABBiOBTIHCAkQABiABNIBCTEwMjI2ajFqN6gCCLACAQ&sourceid=chrome&ie=UTF-8#lrd=0x31cc4b0efd4060e5:0xe81904ed13213c4b,1,,,,'
r = requests.get(link)
print(r)
soup = BeautifulSoup(r.text,'html.parser')
print(soup.prettify())

regex = re.compile('review-snippet')
print(regex)
results = soup.find_all('span', {'class':regex})
# reviews = [result.text for result in results]
print(results)
results[0].text