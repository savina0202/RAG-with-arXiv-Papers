
import arxiv
from sentence_transformers import SentenceTransformer
import pymupdf
import faiss
import numpy as np
import os
import json
from tqdm import tqdm #Progress bar
import pickle #For saving and loading the FAISS index

class ragpipeline:
    def __init__(self, paperlist_filename="paperlist.json", model_name="all-MiniLM-L6-v2"):
        self.paperlist_filename = paperlist_filename
        self.model_name = model_name
        self.paperlist = self.load_paperlist()
        self.model = SentenceTransformer(self.model_name)
        self.index = None  # Initialize FAISS index
        self.chunks = []  
        self.chunk_metadata = []
        self.total_chunks = 0
        self.total_vectors = 0
        self.total_indexed = 0
        # --- File Paths for Saved Data ---
        self.FAISS_INDEX_FILE = "index.faiss"
        self.CHUNKS_FILE = "chunks.pkl"
        self.METADATA_FILE = "metadata.pkl"



        # New: Check for and load existing data on startup
        self.load_index()
        if self.index is None:
            print("No saved index found. The pipeline needs to be built first.")
        else:
            print("FAISS index and data loaded successfully.")


    def load_paperlist(self):
        if os.path.exists(self.paperlist_filename):
            with open(self.paperlist_filename, 'r', encoding="utf-8") as f:
                paperlist = json.load(f)
            return paperlist
        else:
            print(f"Paper list file {self.paperlist_filename} not found.")
            return []
    
    
    def extract_text_from_pdf(self, pdf_path:str) -> str:
        """ Text Extraction: Extract raw text from each PDF. Clean and concatenate the page text into full-document strings"""
        doc = None
        try:
            #Opening a document
            doc = pymupdf.open(pdf_path)
            pages=[]
            for page in doc:
                page_text = page.get_text().strip() # Get raw text from the page
                pages.append(page_text)
            full_text = "\n".join(pages)  # Concatenate all page texts into a single string
            return full_text
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        finally:
            if doc:
                doc.close()


    def chunk_text_sliding_window(self, text:str, max_token:int =512, overlap: int =50) -> list[str]:
        """Chunking Logic (Sliding Windows)"""
        tokens = text.split()  # Simple tokenization by whitespace
        chunks = []
        step = max_token - overlap
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + max_token]
            chunks.append(" ".join(chunk))
            if i + max_token >= len(tokens):
                break
        return chunks
    

    def embadding_text_chunks(self, chunks: list[str]) -> list[tuple[str, list[float]]]:
        """Embedding Logic: Convert text chunks into embeddings using a pre-trained model.
           Sample return format: [("This is chunk 1", embedding1), ("This is chunk 2", embedding2)]
        """
        if not chunks:
            print("No text chunks to embed.")
            return []
        embeddings = self.model.encode(chunks, show_progress_bar=True) # GPU and pyTorch: convert_to_tensor=True  
        return embeddings


    def build_index(self, embeddings: np.ndarray) -> list[str]:
        """Build FAISS Index
            embeddings should be a 2D numpy array of shape (num_chunks, dimension), example: (100, 384)
        """
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)  # FAISS needs to know the dimensionality of the vectors it will be indexing. L2 distance (Euclidean distance)
        self.index.add(embeddings.astype(np.float32))  # Ensure embeddings are in float32 format
        # print(f"FAISS index built with {embeddings.shape[0]} vectors of dimension {dim}.")
        # print(f"Number of vectors in index: {self.index.ntotal}")

    def save_index(self,index_path="faiss_data"):
        """
        Saves the FAISS index, chunks, and metadata to disk.
        """

        if self.index is None:
            print("No index to save.")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(index_path, exist_ok=True)
            
            # Define file paths
            faiss_index_file = os.path.join(index_path, self.FAISS_INDEX_FILE)
            chunks_file = os.path.join(index_path, self.CHUNKS_FILE)
            metadata_file = os.path.join(index_path, self.METADATA_FILE)

            # Save the FAISS index
            faiss.write_index(self.index, faiss_index_file)
            print(f"\nFAISS index saved to {faiss_index_file}")

            # Save the chunks list
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            print(f"Chunks saved to {chunks_file}")

            # Save the chunk metadata
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            print(f"Metadata saved to {metadata_file}")

        except Exception as e:
            print(f"Error saving files: {e}")

        
    def load_index(self, index_path="faiss_data") -> bool:
        """
        Loads the FAISS index, chunks, and metadata from disk if they exist.
        """
        FAISS_INDEX_FILE = os.path.join(index_path, self.FAISS_INDEX_FILE)
        CHUNKS_FILE = os.path.join(index_path, self.CHUNKS_FILE)
        METADATA_FILE = os.path.join(index_path, self.METADATA_FILE)
        
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE) and os.path.exists(METADATA_FILE):
            try:
                # Load the FAISS index
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                
                # Load the chunks list
                with open(CHUNKS_FILE, 'rb') as f:
                    self.chunks = pickle.load(f)

                # Load the chunk metadata
                with open(METADATA_FILE, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                
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
            raise ValueError("FAISS index is not built. Call build_faiss_index() first.")
        
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
        

    def build_rag_runner(self):
        """ Process papers """
       
        if not self.paperlist:
            print("No papers found in the paper list. Please run data clection first.")
            return
        
        print(f"Building RAG from {len(self.paperlist)} PDF files ...")

        idx =1
        all_embeddings = []
        for paper in tqdm(self.paperlist, desc="Processing papers"):

            text = self.extract_text_from_pdf(paper['filename'])
            if text:
                chunks = self.chunk_text_sliding_window(text)
                embeddings = self.embadding_text_chunks(chunks)
                
                # Store chunks and metadata, and append embeddings to a single list
                for i, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.chunk_metadata.append({
                        "paper_title": paper['paper_title'],
                        "filename": paper['filename'],
                        "chunk_index_in_paper": i,
                    })
                all_embeddings.append(embeddings)
                # For statistics
                self.total_chunks += len(chunks)
                self.total_vectors += len(embeddings)
               
                idx += 1  

        if all_embeddings:
            final_embeddings =np.concatenate(all_embeddings, axis=0)  # Concatenate all embeddings into a single array

            # Build the FAISS index with the final embeddings
            self.build_index(final_embeddings)

            # Finally, save the built index and collected data
            self.save_index()
