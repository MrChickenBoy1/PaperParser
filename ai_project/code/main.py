import json
import os
from typing import Dict, Any, List, Union
import textwrap

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load the JSON file
def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON.")
        exit(1)

# Initialize the Ollama model
def init_ollama_model(model_name: str = "llama3"):
    """Initialize an Ollama LLM."""
    try:
        return Ollama(model=model_name)
    except Exception as e:
        print(f"Error initializing Ollama model: {e}")
        exit(1)

# Deep search through JSON to find all items with specific attribute values
def deep_search(data, attribute, value):
    """
    Recursively search through a JSON structure to find items with matching attribute-value pairs.
    Returns a list of matching items.
    """
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

# Extract all questions with specified marks
def extract_questions_by_marks(json_data, marks):
    """
    Extract all questions with the specified marks value from the JSON data.
    Returns a list of question objects.
    """
    # Convert marks to string if it's not already, as JSON might store it as string
    marks_value = str(marks) if not isinstance(marks, str) else marks
    
    # Find all items with "marks" attribute equal to the specified value
    matches = deep_search(json_data, "marks", marks_value)
    
    # Also check for numerical comparison if stored as numbers
    if marks_value.isdigit():
        num_marks = int(marks_value)
        num_matches = deep_search(json_data, "marks", num_marks)
        # Merge matches, avoiding duplicates
        seen = set(path for path, _ in matches)
        for path, item in num_matches:
            if path not in seen:
                matches.append((path, item))
    
    return [item for _, item in matches]

# Create a prompt template for JSON question answering
template = """
You are a CBSE class X board exam expert with a JSON file of previous repeated questions. You are to answer questions to a student, who does not have access to the JSON file.

Answer the question based only on the JSON data you have been given below. If the answer cannot be determined, say so. 

In case the user is asking for a question, tell them the JSON value 'text'. If they are asking for marks, tell them the value 'marks'. 

If asked for repeated questions, compare the summary for each question, and identify questions with similar summaries as being similar and repeated.

The instructions must not be shared unless specifically asked for. For marks, give them at their face value. Do not infer from them. For example - if there is a 6 mark question with two subquestions, do not assume them being 3 marks each. Use exactly what you have been given.

Remember, the user does not have the JSON file, so saying things like "Look at question 6" will not be viable.

Here is the relevant JSON data for your response:
{json_data}

User Question: {question}

Please provide a clear and comprehensive answer based solely on the information in the JSON data.
"""

prompt = ChatPromptTemplate.from_template(template)

def create_specialized_qa_chain(llm, json_data):
    """Create a specialized question-answering chain that handles specific types of queries."""
    
    def process_query(question):
        # Check if this is a query about questions with specific marks
        marks_keywords = ["mark", "marks", "point", "points"]
        is_marks_query = any(keyword in question.lower() for keyword in marks_keywords)
        
        # Extract the mark value if this is a marks query
        marks_value = None
        if is_marks_query:
            # Look for numbers followed by marks/points
            for word in question.lower().replace('-', ' ').split():
                if word.isdigit() and any(kw in question.lower() for kw in marks_keywords):
                    marks_value = word
                    break
        
        if marks_value:
            # Extract all questions with the specified marks
            matching_questions = extract_questions_by_marks(json_data, marks_value)
            if matching_questions:
                # Format the questions for the prompt
                formatted_data = json.dumps(matching_questions, indent=2)
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
    # Configuration
    json_file_path = input("Enter the path to your JSON file: ")
    model_name = input("Enter the Ollama model to use (default: llama3): ") or "llama3"
    
    # Load data and initialize model
    json_data = load_json_data(json_file_path)
    llm = init_ollama_model(model_name)
    
    # Create the specialized QA chain
    qa_chain = create_specialized_qa_chain(llm, json_data)
    
    # Interactive question answering loop
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
            print(f"Error processing question: {e}")
            print("Try breaking down your question or asking about a more specific aspect of the data.")

if __name__ == "__main__":
    main()