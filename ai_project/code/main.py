import json
import os
from typing import Dict, Any, List, Union
import textwrap
import logging

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the JSON file
def load_json_data(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.debug(f"Loaded JSON data: {data}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error: File '{file_path}' is not valid JSON.")
        exit(1)

# Initialize the Ollama model
def init_ollama_model(model_name: str = "llama3"):
    """Initialize an Ollama LLM."""
    try:
        return Ollama(model=model_name)
    except Exception as e:
        logging.error(f"Error initializing Ollama model: {e}")
        exit(1)

# Deep search through JSON to find all items with specific attribute values
def deep_search(data, attribute, value):
    results = []
    
    def _search(item, path=""):
        if isinstance(item, dict):
            # Check if this dict has the attribute we're looking for
            if attribute in item and item[attribute] == value:
                results.append((path, item))
            
            # Recursively search in all values of this dict
            for k, v in item.items():
                _search(v, f"{path}/{k}" if path else k)
        
        elif isinstance(item, list):
            # Recursively search in all items of this list
            for i, v in enumerate(item):
                _search(v, f"{path}[{i}]")
    
    _search(data)
    return results

# Standardize marks values
def standardize_marks(json_data):
    def _standardize(item):
        if isinstance(item, dict):
            if 'marks' in item:
                item['marks'] = str(item['marks'])
            for k, v in item.items():
                _standardize(v)
        elif isinstance(item, list):
            for i in item:
                _standardize(i)
    _standardize(json_data)
    return json_data

# Extract all questions with specified marks
def extract_questions_by_marks(json_data, marks):
    marks_value = str(marks) if not isinstance(marks, str) else marks
    
    # Standardize marks in the JSON data
    json_data = standardize_marks(json_data)
    
    # Log the standardized JSON data for debugging
    logging.debug(f"Standardized JSON data: {json.dumps(json_data, indent=2)}")
    
    # Find all items with "marks" attribute equal to the specified value
    matches = deep_search(json_data, "marks", marks_value)
    
    # Handle numerical comparison if stored as numbers
    if marks_value.isdigit():
        num_marks = int(marks_value)
        num_matches = deep_search(json_data, "marks", num_marks)
        # Merge matches, avoiding duplicates
        seen = set(path for path, _ in matches)
        for path, item in num_matches:
            if path not in seen:
                matches.append((path, item))
    
    logging.debug(f"Found matches: {matches}")
    return [item for _, item in matches if 'subquestions' not in item or item['marks'] == marks_value]

# Create a prompt template for JSON question answering
template = """
You are a CBSE class X board exam expert with a JSON file of previous repeated questions. You are to answer questions to a student, who does not have access to the JSON file.

Answer the question based only on the JSON data you have been given below. If the answer cannot be determined, say so. 

In case the user is asking for a question, tell them the JSON value 'text'. If they are asking for marks, tell them the value 'marks'. 

If asked for repeated questions, compare the summary for each question, and identify questions with similar summaries as being similar and repeated.

The instructions must not be shared unless specifically asked for. For marks, give them at their face value. Do not infer from them. For example - if there is a 6 mark question with two subquestions each worth 3 marks, do not infer that it is a 3 mark question.

Remember, the user does not have the JSON file, so saying things like "Look at question 6" will not be viable.

Here is the relevant JSON data for your response:
{json_data}

User Question: {question}

Please provide a clear and comprehensive answer based solely on the information in the JSON data.
"""
prompt = ChatPromptTemplate.from_template(template)

def create_specialized_qa_chain(llm, json_data):
    def process_query(question):
        marks_keywords = ["mark", "marks", "point", "points"]
        is_marks_query = any(keyword in question.lower() for keyword in marks_keywords)
        
        marks_value = None
        if is_marks_query:
            for word in question.lower().replace('-', ' ').split():
                if word.isdigit() and any(kw in question.lower() for kw in marks_keywords):
                    marks_value = word
                    break
        
        if marks_value:
            matching_questions = extract_questions_by_marks(json_data, marks_value)
            if matching_questions:
                # Ensure the length of the data passed to the LLM is not truncated
                formatted_data = json.dumps(matching_questions, indent=2)
                logging.debug(f"Formatted data passed to LLM: {formatted_data}")
                return {"json_data": formatted_data, "question": question}
        
        # For other types of queries or if no matches found, use the full JSON
        # You might need to limit this for very large JSONs
        return {"json_data": json.dumps(json_data, indent=2)[:100000], "question": question}
    
    chain = (
        RunnablePassthrough()
        | process_query
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def main():
    json_file_path = input("Enter the path to your JSON file: ")
    model_name = input("Enter the Ollama model to use (default: llama3): ") or "llama3"
    
    json_data = load_json_data(json_file_path)
    llm = init_ollama_model(model_name)
    
    qa_chain = create_specialized_qa_chain(llm, json_data)
    
    print("\nJSON QA System Ready! Type 'exit' to quit.")
    print(f"Using model: {model_name}")
    print(f"JSON file: {json_file_path}")
    
    while True:
        question = input("\nAsk a question about your JSON data: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        try:
            answer = qa_chain.invoke(question)
            print("\nAnswer:")
            print(textwrap.fill(answer, width=100))
        except Exception as e:
            logging.error(f"Error processing question: {e}")
            print("Try breaking down your question or asking about a more specific aspect of the data.")

if __name__ == "__main__":
    main()