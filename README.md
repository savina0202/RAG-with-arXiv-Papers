
# Retrieval-Augmented Generation (RAG) with arXiv Papers

RAG architectures allow the agent to search a structured knowledge base and generate grounded, document-aware answers. This RAG pipeline uses recent arXiv cs.CL papers, converting them into searchable chunks, embedding and indexing them with FAISS. A query interface is provided that takes the user's question, retrieves the top relevant chunks and displays them for further processing. 

Starting to build a private research knowledge base, and planning to be replaced by a full-featured hybrid database.

**Deliverables**

<img width="945" height="302" alt="image" src="https://github.com/user-attachments/assets/87f744d7-e374-4b92-8e44-91aed730568e" />

1. **Code Notebook / Script:** Complete code for the RAG pipeline (PDF extraction, chunking, embedding, indexing, retrieval).<br><br>
   **core/ragpipeline.py**<br>
   **ragpipeline.ipynb**

2. **Data & Index:** The FAISS index file and the set of 50 processed paper chunks (e.g., as JSON or pickled objects).<br><br>
   **papers/...**<br>
   **faiss_data/...**<br>

3. **Retrieval Report:** A brief report showing at least 5 example queries and the top-3 retrieved passages for each, to demonstrate system performance.<br><br>
    **test_search_api.py**<br>
    **search_api_test_responses.json**<br>

4. **FastAPI Service:** The FastAPI app code (e.g. main.py) and instructions on how to run it. The /search endpoint should be demonstrable (e.g. returning top-3 passages in JSON for sample queries).<br><br>
    **main.py**<br>

<img width="1174" height="790" alt="image" src="https://github.com/user-attachments/assets/3db0f192-cf11-4908-b83f-7d54e47808c3" />
<img width="1160" height="796" alt="image" src="https://github.com/user-attachments/assets/ba1237c7-5d75-4b2c-8044-b081ae4d8868" />


