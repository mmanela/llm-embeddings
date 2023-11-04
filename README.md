# Exploring LLM Embeddings

## Blog post
Read my [blog post](https://matthewmanela.com/blog/exploring-the-power-of-llm-embeddings) which describes the details of this project.  

## Built with
- Python 3.11.1
- pip 23.2.1

  
## Setup


First install Python dependencies
```
cd app
pip install -r requirements.txt
```

Then download NLTK models 

```python
import nltk
nltk.download('punkt')

# This is for lemmatization
nltk.download('wordnet')
```

Then add a .env file to the root with your OPENAI_API_KEY
```
OPENAI_API_KEY=YOU_KEY_HERE
```


## Mode

- **create** - Creates the embeddings model for a given file
- **query** - Queries the document's embeddings model with a given query to see what it is most similar to
- **extract** - Helps debug the doc extraction logic by showing how the given file is split up
- **test** - Explores comparing 3 pieces of text to learn how similar or different they are based on their embeddings
- **create_dict** - Creates an embeddings model for a list of words
- **query_dict** - Queries the dictionary's embeddings model with a given query to see what it is most similar to
- **analyze** - Performance analysis, clustering, categorization on the document model (must first run create and create_dict)


## Examples

**Explore the parsing and splitting of an input file**
```
./run.sh -m extract -f pair_programming.pdf           
```


**Create the embeddings model for a file**
```
./run.sh -m create -f pair_programming.pdf           
```

**Create a model for a pre-defined word list**
```
./run.sh -m create_dict
```

**Query the dictionary word model **
```
./run.sh -m query_dict -q "hotdogs are the best food"
```

**Analyze the embeddings the file leveraging the embeddings model and dictionary model**
This leads to unexpected results which may mean simply negating the embeddings is not enough to get the opposite meaning 
```
./run.sh -m analyze -f pair_programming.pdf               
```


**Query the model so see what is similar to the query**
```
./run.sh -m query -q "absolute correctness is an illusion"  -f pair_programming.pdf     
```

**Compare embeddings in general to see similarity**
```
./run.sh -m test -t1 "people are bad and should not be trusted" -t2 "hotdogs are the best food" -t3 "I hate everyone" 


./run.sh -m test -t1 "UFOs are real and have visited earth on many occasions" -t2 "Bigfoot lives in my backyard" -t3 "the NY Jets have had a long losing streak" 

```

Query a single word in the dict and see the most similar and different words based on its embedding
```
./run.sh -m query_word -q eating  
```

## Notes
 

### SPACY
If you want to use spacy instead of NLTK for text splitting you must download it's language model
```
python -m spacy download en_core_web_sm
```
