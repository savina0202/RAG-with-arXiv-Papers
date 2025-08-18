# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add the parent directory to the Python path to allow importing from 'core'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.ragepipeline import ragpipeline

# Define a Pydantic model for the request body to ensure proper data validation
class SearchRequest(BaseModel):
    query: str
    k: int = 3 # Default value for k is 3 if not provided

# Initialize the FastAPI application
app = FastAPI(title="RAG Search API", description="An API to search a document corpus using a RAG pipeline.")

# Initialize the ragpipeline instance
# This object will be loaded into memory once when the application starts
try:
    rag_pipeline = ragpipeline()
    if rag_pipeline.index is None:
        raise RuntimeError("RAG pipeline index not found. Please ensure it is built and saved.")
except Exception as e:
    # If the pipeline can't be initialized, the API should not start
    print(f"Failed to initialize RAG pipeline: {e}")
    sys.exit(1)


@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """
    Search the indexed documents for the most relevant chunks.
    
    - **query**: The search query string.
    - **k**: The number of top results to return (default: 3).
    """
    try:
        results = rag_pipeline.search(request.query, request.k)
        return {"query": request.query, "results": results}
    except ValueError as e:
        # Catch the specific error from the ragpipeline search function
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# If you run this script directly, start the uvicorn server
if __name__ == "__main__":
    # The host '0.0.0.0' makes the server accessible from outside the local machine (if needed).
    # The port 8000 is the standard for FastAPI.
    # The 'reload=True' flag is useful for development as it restarts the server on code changes.
    uvicorn.run(app, host="0.0.0.0", port=8000)