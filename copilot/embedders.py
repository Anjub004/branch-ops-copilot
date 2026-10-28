"""Turn text into vectors. Two interchangeable backends:

- SpacyEmbedder  : averaged word vectors from spaCy's en_core_web_md. Fully offline, fast, modest quality.
- OllamaEmbedder : a real embedding model (default nomic-embed-text) through any OpenAI-compatible
                   /v1/embeddings endpoint. Better quality; needs `ollama pull nomic-embed-text`.

Both return L2-normalised numpy arrays, so a dot product is cosine similarity.
"""
import os
import numpy as np


def _normalise(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SpacyEmbedder:
    name = "spacy-word-vectors"

    def __init__(self, model="en_core_web_md"):
        import spacy
        self.nlp = spacy.load(model, disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])

    def embed(self, texts):
        vectors = []
        for doc in self.nlp.pipe(texts):
            words = [t.vector for t in doc if t.has_vector and t.is_alpha and not t.is_stop]
            vectors.append(np.mean(words, axis=0) if words else np.zeros(self.nlp.vocab.vectors_length))
        return _normalise(vectors)


class OllamaEmbedder:
    def __init__(self, model=None):
        from copilot.llm import client
        self.client = client
        self.model = model or os.getenv("EMBED_MODEL", "nomic-embed-text")
        self.name = f"ollama:{self.model}"

    def embed(self, texts):
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return _normalise([item.embedding for item in response.data])


def get_embedder(kind=None):
    kind = kind or os.getenv("EMBEDDER", "spacy")
    return OllamaEmbedder() if kind == "ollama" else SpacyEmbedder()
