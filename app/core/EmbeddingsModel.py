import numpy as np
from langchain import FAISS
import pickle
import json
import os
import pickle
from core.EmbeddedReference import EmbeddedReference
from core.getEmbeddingsProvider import getEmbeddingsProvider

root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
VECTOR_STORE_PATH = os.path.join(root_dir, "_cache", "_vector_store")
TEXT_STORE_PATH = os.path.join(root_dir, "_cache", "_text_store")


class EmbeddingsModel(object):
    def __init__(self, provider, fileName) -> None:
        self.provider = provider
        self.fileName = fileName
        self.name = os.path.splitext(fileName)[0]

        # Ensure text store path is created
        os.makedirs(TEXT_STORE_PATH, exist_ok=True)

    def _get_paths_for_storage(self):
        return dict(
            qualifiedName=f'{self.provider}_{self.name}',
            embeddingStorePath=os.path.join(
                TEXT_STORE_PATH, f'{self.provider}_{self.name}.pickle'),
            textStorePath=os.path.join(
                TEXT_STORE_PATH, f'{self.provider}_{self.name}.json'),
            vectorStoreFaiss=os.path.join(
                VECTOR_STORE_PATH, self.name + '.faiss'),
            vectorStorePkl=os.path.join(
                VECTOR_STORE_PATH, self. name + '.pkl'),
        )

    def _getSavedEmbeddingReferences(self, mapPath):
        embedding_objects: list[EmbeddedReference] = None
        with open(mapPath, 'rb') as f:
            embedding_objects = pickle.load(f)

        embeddingsArr = np.array([x.embedding for x in embedding_objects])
        return embedding_objects, embeddingsArr

    def load_model(self, indexOnly: bool = False):
        paths = self._get_paths_for_storage()
        cached_embedder = getEmbeddingsProvider(self.provider)
        faiss_index = FAISS.load_local(
            folder_path=VECTOR_STORE_PATH, index_name=paths.get('qualifiedName'), embeddings=cached_embedder)

        if indexOnly:
            return faiss_index, None, None
        else:
            embedding_objects, embeddings_list = self._getSavedEmbeddingReferences(
                paths.get('embeddingStorePath'))

            return faiss_index, embedding_objects, embeddings_list

    def build_and_persist_model(self, texts: list[str], metadata=None):

        paths = self._get_paths_for_storage()

        if os.path.exists(os.path.join(VECTOR_STORE_PATH, self.name)):
            print(f'Removing existing store {self.name}')
            os.remove(paths.get('vectorStoreFaiss'))
            os.remove(paths.get('vectorStorePkl'))

        cached_embedder = getEmbeddingsProvider(self.provider)

        doc_embeddings = cached_embedder.embed_documents(texts)
        embedding_objects: list[EmbeddedReference] = []
        for i, embedding in enumerate(doc_embeddings):
            embedding_objects.append(EmbeddedReference(
                embedding=embedding, metadata=metadata[i] if metadata is not None else None, content=texts[i]))
        with open(paths.get('embeddingStorePath'), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(embedding_objects, f, pickle.HIGHEST_PROTOCOL)

        # Store the data in a text format for debugging
        with open(paths.get('textStorePath'), 'w', encoding='utf-8') as f:
            json.dump([x.forJson() for x in embedding_objects], f,
                      ensure_ascii=False, indent=4)

        faiss_index = FAISS.from_texts(texts, cached_embedder)
        faiss_index.save_local(VECTOR_STORE_PATH, paths.get('qualifiedName'))

