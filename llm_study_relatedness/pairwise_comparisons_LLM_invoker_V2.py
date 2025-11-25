#!/usr/bin/python3.5

import requests
import json
from pathlib import Path
import sys
import io
from collections import OrderedDict

OLLAMA_INSTANCES = 4

# Reconfigure sys.stdout to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def ask_question(ollama_url, model_name, study_txt1, study_txt2):
    """Sends a structured prompt to the model and ensures JSON response."""
    headers = {"Content-Type": "application/json"}

    # Enforce structured JSON response
    structured_prompt = [
        "How high similarity do you expect between these two microbiome studies? Instructions: focus on the biomes from which the samples were collected. In the next line quantify the similarity strictly according to these categories: '***high***' (studies of very similar biomes from the same organisms, closely related environments, and similar conditions), '***medium***' (studies of similar biomes from the same organisms or similar environments), '***low***' (studies of the same type of biome but from different organisms or environments) or '***no***' (completely unrelated studies, such as a host microbiome and a soil microbiome).\n\nStudy 1: ",
        study_txt1,
        "\n\nStudy 2: ",
        study_txt2,
        "\n\nPlease provide your response in JSON format with the following structure:\n{{\n    \"explanation\": \"<short explanation>\",\n    \"answer\": \"<***high*** or ***medium*** or ***low*** or ***no***>\"\n}}\nOnly return a valid JSON object."
    ]

    payload = {
        "model": model_name,
        "prompt": "".join(structured_prompt),
        "format": "json",
        "stream": False
    }
    
    try:
        response = requests.post(ollama_url, headers=headers, json=payload, stream=False)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print("Request error: ", e)
        return {"error": "Request failed."}

    if not response:
        print("Empty response from the model.")
        return {"error": "Empty response from model"}

    try:
        parsed_response = response.json()  # Ensure response is valid JSON
        if isinstance(parsed_response, dict) and "response" in parsed_response:
            parsed_response = json.loads(parsed_response["response"])
            if "explanation" in parsed_response and "answer" in parsed_response:
                return OrderedDict([
                    ("explanation", parsed_response["explanation"]),
                    ("answer", parsed_response["answer"])
                ])
            else:
                return {"error": "Unexpected JSON structure from model response"}
        else:
            return {"error": "Unexpected JSON structure from model"}
    except json.JSONDecodeError as e:
        print("Failed to parse final response as JSON: ", e)
        print(response.text)
        return {"error": "Invalid JSON response from model"}

def process_studies(ollama_url, model_name, studies_file_path, study_pairs_file_path, output_file_path, process_index, process_size):
    """Processes studies and saves JSON responses."""
    
    results = []  # List to store structured output
    
    studies = {}
    with studies_file_path.open('r', encoding='utf-8') as input_file:
        for line in input_file:
            study_id, study_txt = line.strip().split("\t", 1)
            studies[study_id] = study_txt

    with study_pairs_file_path.open('r', encoding='utf-8') as input_file:
        pair_index = 0
        for line in input_file:
            study_id1, study_id2, score = line.strip().split("\t")
            if study_id1 in studies and study_id2 in studies:
                study_txt1 = studies[study_id1]
                study_txt2 = studies[study_id2]
                if pair_index % process_size == process_index:
                    response_json = ask_question(ollama_url, model_name, study_txt1, study_txt2)
                    results.append(OrderedDict([
                        ("Study_ID1", study_id1),
                        ("Study_ID2", study_id2),
                        ("Response", response_json)
                    ]))
                pair_index += 1
                    
    with output_file_path.open('w', encoding='utf-8') as output_file:
        json.dump(results, output_file, indent=4, ensure_ascii=False)
    
    print("JSON output saved to: ", output_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 5 and len(sys.argv) != 7:
        print("Usage: ", sys.argv[0], " <model_name> <studies_file> <study_pairs_file> <output_file> [process_index] [process_size]")
        sys.exit(1)

    model_name = sys.argv[1]
    studies_file_path = Path(sys.argv[2])
    study_pairs_file_path = Path(sys.argv[3])
    output_file_path = Path(sys.argv[4])
    process_index = 0
    process_size = 1
    if len(sys.argv) == 7:
        process_index = int(sys.argv[5])
        process_size = int(sys.argv[6])

    ollama_port = 11434 + process_index % OLLAMA_INSTANCES
    ollama_url = "http://localhost:"+str(ollama_port)+"/api/generate"

    if not studies_file_path.exists():
        print("Error: Studies file does not exist: ", studies_file_path)
        sys.exit(1)

    if not study_pairs_file_path.exists():
        print("Error: Studies file does not exist: ", study_pairs_file_path)
        sys.exit(1)

    if not output_file_path.parent.exists():
        print("Error: Output directory does not exist: ", output_file_path.parent)
        sys.exit(1)

    process_studies(ollama_url, model_name, studies_file_path, study_pairs_file_path, output_file_path, process_index, process_size)
