import argparse
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter
import os
import dotenv
import numpy as np 
import pandas as pd
from scipy.spatial import distance
from core.DocumentAnalyzer import DocumentAnalyzer
from core.EmbeddingsModel import EmbeddingsModel
from core.getEmbeddingsProvider import getEmbeddingsProvider

# loading the .env file
dotenv.load_dotenv()


FILES_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir, "files")
 
WORDS_DICT_NAME = "scowl10to70_lemmatization"
WORDS_FILE_PATH = os.path.join(FILES_FOLDER, f"{WORDS_DICT_NAME}.txt")

SNIPPET_LENGTH = 100

def parse_pdf(filename):
    path = os.path.join(FILES_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f'File {path} not found')
 
    chunk_size = 250
    chunk_overlap = 0
    loader = PyPDFLoader(path)
    # spacy_splitter = SpacyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # recursive_splitter = RecursiveCharacterTextSplitter(  chunk_size = chunk_size,  chunk_overlap  = chunk_overlap)
    nltk_splitter = NLTKTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = loader.load_and_split(text_splitter=nltk_splitter)
    print(f'Loaded {len(texts)} texts')
    return texts


def main():

    args = parser.parse_args()
    filename = args.filename
    provider = args.provider
    name = os.path.splitext(filename)[0] if filename is not None else None

    if args.mode == 'extract':
        print(f'Exploring splitting of {filename}')
        texts = parse_pdf(filename)
        for n in range(0, 10):
            print(f'\n[SNIPPET {n}]: \n {texts[n].page_content[0:SNIPPET_LENGTH]}')

    elif args.mode == 'create':
        print(f'Creating new embeddings into store {name}')
        texts = parse_pdf(filename)

        contents = [text.page_content for text in texts]
        metadata = [text.metadata for text in texts]

        docModel = EmbeddingsModel(provider, filename)
        docModel.build_and_persist_model(contents, metadata)

    elif args.mode == 'analyze':
        print(f'Loading existing store {name} and analyzing')

        analyzer = DocumentAnalyzer(provider, filename, WORDS_DICT_NAME)
        df = analyzer.get_summary_data_frame()
        print(df.to_string())

        analyzer.render_histogram()
        analyzer.render_cluster_chart()
        analyzer.show_charts()

    elif args.mode == 'query':
        query = args.query
        print(f'Querying existing store {name} with query {query}')

        docModel = EmbeddingsModel(provider, filename)
        faiss_index, embedding_objects, embeddings_list = docModel.load_model()

        docs_and_scores = faiss_index.similarity_search_with_score(query, 3)
        snippet_and_score = [(x[0].page_content[0:SNIPPET_LENGTH], x[1])
                             for x in docs_and_scores if x[0].page_content]
        print(snippet_and_score)

    elif args.mode == 'create_dict':
        print(f'Creating dictionary words for embedding store')

        words = []

        with open(WORDS_FILE_PATH, 'r') as f:
            words = [line.rstrip() for line in f]

        print(f'Loaded {len(words)} words')

        wordModel = EmbeddingsModel(provider, WORDS_DICT_NAME)
        wordModel.build_and_persist_model(words)

    elif args.mode == 'query_dict':
        query = args.query
        print(
            f'Querying dictionary store {WORDS_DICT_NAME} with query {query}')

        wordModel = EmbeddingsModel(provider, WORDS_DICT_NAME)
        words_index, _, _ = wordModel.load_model(True)

        docs_and_scores = words_index.similarity_search_with_score(query, 3)
        snippet_and_score = [(x[0].page_content[0:SNIPPET_LENGTH], x[1])
                             for x in docs_and_scores if x[0].page_content]
        print(snippet_and_score)

    elif args.mode == 'test':
        test1: str = args.test1
        test2: str = args.test2
        test3: str = args.test3
        print(
            f'Testing with\n\ntest1\n\t{test1}\n\ntest2\n\t{test2}\n\ntest3\n\t{test3}\n\n')
        cached_embeddings = getEmbeddingsProvider(provider)
        embeddings = cached_embeddings.embed_documents([test1, test2, test3])

        arr1 = np.array(embeddings[0])
        arr2 = np.array(embeddings[1])
        arr3 = np.array(embeddings[2])
        cosine_distance_1_2 = distance.cosine(arr1, arr2)
        euclidean_distance_1_2 = distance.euclidean(arr1, arr2)
        cosine_distance_1_3 = distance.cosine(arr1, arr3)
        euclidean_distance_1_3 = distance.euclidean(arr1, arr3)
        cosine_distance_2_3 = distance.cosine(arr2, arr3)
        euclidean_distance_2_3 = distance.euclidean(arr2, arr3)

        dist_frame = pd.DataFrame({
            "tests": ["test1 vs test2", "test1 vs test3", "test2 vs test3"],
            "cosine_distance": [cosine_distance_1_2, cosine_distance_1_3, cosine_distance_2_3],
            "euclidean_distance": [euclidean_distance_1_2, euclidean_distance_1_3, euclidean_distance_2_3],
        })

        print(dist_frame)
 

parser = argparse.ArgumentParser(
    prog='Embedding Exploration',
    description='Explore the world of emdeddings')
parser.add_argument('-f', '--filename', required=False)
parser.add_argument(
    '-m', '--mode', choices=['create', 'analyze', 'query', 'extract', 'test', 'create_dict', 'query_dict'])
parser.add_argument('-p', '--provider', choices=['openai'], default='openai')
parser.add_argument('-q', '--query', required=False)
parser.add_argument('-t1', '--test1', required=False)
parser.add_argument('-t2', '--test2', required=False)
parser.add_argument('-t3', '--test3', required=False)


if __name__ == "__main__":
    main()
