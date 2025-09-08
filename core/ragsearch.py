from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle #For saving and loading the FAISS index
from core.config import (
    FAISS_INDEX_FILE,
    CHUNKS_FILE,
    METADATA_FILE
)

class ragsearch:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []  # List to store text chunks
        self.chunk_metadata = []  # List to store metadata for each chunk (e.g., paper title, filename, chunk index)
        self.index = None
        self.load_index()

    def load_index(self) -> bool:
        """
        Loads the FAISS index, chunks, and metadata from disk if they exist.
        """
        try:
            if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(METADATA_FILE):

                # Load the FAISS index
                self.index = faiss.read_index(str(FAISS_INDEX_FILE))
                #print(f"loaded successfully")
                # Load the chunks list
                with open(CHUNKS_FILE, 'rb') as f:
                    self.chunks = pickle.load(f)

                # Load the chunk metadata
                with open(METADATA_FILE, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                #print(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                return True
        except Exception as e:
            print(f"Error loading saved files: {e}")
            self.index = None
            self.chunks = []
            self.chunk_metadata = []
            return False
        return False

    def search(self, query: str, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for the k nearest neighbors of the query embedding.
            Returns distances and indices of the nearest neighbors.
        """
        if self.index is None:
            self.load_index()
            #raise ValueError("FAISS index is not built. Call build_faiss_index() first.")
        
        # Get query_embedding from the text
        query_embedding = self.model.encode([query], show_progress_bar=False)  # Encode the query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[0] != 1:
            raise ValueError("Query embedding should be a single vector, but got shape: {}".format(query_embedding.shape))
        if query_embedding.shape[1] != self.index.d:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} does not match index dimension {self.index.d}.")
        
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, k)  # Search the index
        print(f"Search completed. Found {len(distances[0])} nearest neighbors.")

        result=[]
        for indice, distance in zip(indices[0], distances[0]):
            result.append({
                "distance": float(distance),
                "chunk": self.chunks[indice],
                "metadata":self.chunk_metadata[indice]
            })

        return result
    
# if __name__ == "__main__":
#     ragsearch= ragsearch()
#     if ragsearch.load_index():
#         print("Index loaded successfully.")
#     else:
#         print("Failed to load index.")
    
#     query = "What is the role of attention mechanisms in transformer models?"
#     results = ragsearch.search(query, k=3)  
#     for res in results:
#         print(f"Distance: {res['distance']}")
#         print(f"Chunk: {res['chunk']}")
#         print(f"Metadata: {res['metadata']}")
#         print("-----")
