# test_api_search.py
import requests
import json
import os

# Define the base URL of your FastAPI application.
# Make sure the server is running on this host and port.
API_URL = "http://127.0.0.1:8000"

def test_search_endpoint(query: str, k: int):
    """
    Sends a POST request to the /search endpoint of the API and handles the response.

    Args:
        query (str): The search query to send to the API.
        k (int): The number of results to request.
    """
    endpoint = f"{API_URL}/search"
    payload = {
        "query": query,
        "k": k
    }

    print(f"Sending request to {endpoint} with payload: {payload}")

    try:
        # Send the POST request with a JSON payload
        response = requests.post(endpoint, json=payload, timeout=10)

        # Check the status code of the response
        if response.status_code == 200:
            print(f"Success! Status Code: {response.status_code}")
            try:
                # Get the JSON response data
                response_data = response.json()
                return response_data

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON response from server. Status Code: {response.status_code}")
                print(f"Response content: {response.text}")
                return None
        else:
            print(f"Error! Status Code: {response.status_code}")
            print(f"Error details: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle network or connection-related errors
        print(f"Failed to connect to the API server: {e}")

if __name__ == "__main__":
    # Example usage with your query

    search_queries = ["What are transformers in NLP?",
                      "Can LLMs serve as simulator of world knowledge?",
                      "how to enhance automated interpreting accessment with explainable AI?",
                      "how to enhance both reasoning capabilities and performance in empathy and expertise?",
                      "what is the result of human evaluation between VAC and the best performing baseline?"
                      ]
    all_resoonses = []
    number_of_results = 3
    for search_query in search_queries:
        response_data=test_search_endpoint(search_query, number_of_results)
        if response_data:
            all_resoonses.append(response_data)


    if all_resoonses:
        # Save the responses to a JSON file
        output_file = "search_api_test_responses.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_resoonses, f, indent=4)
        print(f"Search responses saved to {output_file}")
    else:
        print("No valid responses received from the API.")