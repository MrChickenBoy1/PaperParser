from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import uuid
from qdrant_client.models import PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import streamlit as stl
from langchain_community.document_loaders import JSONLoader
import json
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

print("Loading...")
with open('ai_project\code\question‑schema.json', 'r') as f:
    json_schema = json.load(f)

print("Loading JSON file...")


loader = JSONLoader(
    file_path="ai_project\Subjects\english\papers\extracted_questions.json",
    jq_schema=".[].question",
    text_content=False,
)

docs = loader.load()

loader = JSONLoader(
    file_path="ai_project\Subjects\english\papers\extracted_questions.json",
    jq_schema="del(.[].question)",
    text_content=False,
)

metadata = loader.load()




for i in metadata:
    data_list = json.loads(i.page_content)



increment = 0 
for i in docs:
    i.metadata = data_list[increment]
    increment += 1

print("Loaded Files!")

llm = ChatOllama(model="llama3")



client = QdrantClient(path="ai_project") 
print("Adding to Vector database...")


client.delete_collection(collection_name="questions")
client.create_collection(
    collection_name="questions",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

points = []

for doc in docs:
    text = doc.page_content
    meta = doc.metadata
    vector = embeddings.embed_query(text)  # Ollama embedding
    payload = meta.copy()
    payload["text"] = text

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload 
    )
    points.append(point)

client.upsert(
    collection_name="questions",
    points=points
)

print("Added to Vector database!")

while True:
    schema_str = json.dumps(json_schema, indent=2)
    query = str(input("Enter your query: "))

    intent = llm.invoke(f"""
                        You are a classifier that analyzes student queries and classifies their intent for a Class 10 board exam assistant.

    The assistant can perform two actions:
    1. **retrieval** – Fetch specific past year questions from a database.
    2. **advisory** – Provide study guidance, strategies, or trends (e.g., what’s important, how questions are asked).

    ---

    ### Your job:

    Given a user query, classify its intent as:

    - `"retrieval"` → if the user is clearly asking for past questions based on subject, chapter, marks, type, or repetition.
    - `"advisory"` → if the user is asking for help with what to study, what’s important, how topics are asked, or any general advice.
    - `"both"` → if the query mixes both, e.g., asking for important questions from a topic.
    - `"unknown"` → if the intent is unclear or not related to exams.

    ---

    ### Examples:

    - Query: *"Give me 5-mark questions from Science"* → **retrieval**
    - Query: *"What are the most repeated questions in SST?"* → **retrieval**
    - Query: *"What kind of questions are usually asked in English?"* → **advisory**
    - Query: *"I didn’t study anything, what should I focus on?"* → **advisory**
    - Query: *"Give me important questions from Civics"* → **both**
    - Query: *"How are you?"* → **unknown**

    ---

    Now classify this query and return only one word from: `retrieval`, `advisory`, `both`, or `unknown`.  
    Query: {query}
    """)

    print("Intent classified as: ", intent.content)

    if (intent.content) == "**retrieval**" or (intent.content) == "`retrieval`":
        subject = llm.invoke(f"""
                            You are a word analyser. If the sentence I give you, has at most one of the word: 
                            maths, english, social science, science, computer, hindi; 
                            then output the word. If the sentence does not contain any of these words, then output 'all' and if it has more than one of these words, output 'error'.                               
                            Give the answer ONLY in one word. The sentence is: {query}""")

        if subject.content != "all":
            subject_list=open(f"ai_project\Subjects\{subject.content}\Chapters.txt", "r")
            keyword_list_open = open(f"ai_project\Subjects\{subject.content}\keywords.txt", "r")
        elif subject.content == "error":
            print("Enter a valid query!")
            continue
        else:
            subject_list = ""
            keyword_list_open = ""
        chapters = ""
        keyword_list = ""

        for i in subject_list:
            chapters += (i+"\n")

        for j in keyword_list_open:
            keyword_list += (j+"\n")

        system_prompt = f"""
        You are a JSON filter generator for a database query system. Follow these instructions EXACTLY:

        ==== CRITICAL RULES (VIOLATION NOT ALLOWED) ====

        1. Your output must be ONLY valid JSON - no explanations, no comments, no markdown.

        2. NEVER INCLUDE keys with these values:
        - null
        - false 
        - []
        - ""
        - "unknown"
        - "any"

        3. If information is not in the query:
        - OMIT THE ENTIRE KEY from your JSON
        - DO NOT set the value to null, false, or empty array
        
        4. DO NOT INFER OR ASSUME information not in the query.

        5. DO NOT include ANY FIELD that isn't explicitly mentioned or clearly implied in the query.

        ==== SPECIFIC FIELD RULES ====

        - "subject": Always use "{subject.content}" (unless subject is "all", then OMIT this key)
        - "chapter": Only include if a specific chapter is mentioned in these: {chapters}
        - "question_type": ONLY include if the query EXPLICITLY contains one of: "extract-based", "very short answer", "short answer", "long answer"
        - "marks": ONLY include if a specific number is mentioned with "marks", "mark", or "points"
        - "is_repeated": ONLY include (with value true) if words like "repeated", "frequent", "common", "important" appear
        - "keywords": ONLY include if specific keywords other than chapter names are mentioned

        ==== EXAMPLES ====

        1. Query: "questions about coorg from english"
        CORRECT: {{"subject": "english", "chapter": "glimpses of india"}}
        WRONG: {{"subject": "english", "chapter": "glimpses of india", "marks": null, "question_type": null}}

        2. Query: "5 mark questions from Nelson Mandela"
        CORRECT: {{"subject": "english", "chapter": "nelson mandela", "marks": 5}}
        WRONG: {{"subject": "english", "chapter": "nelson mandela", "marks": 5, "question_type": null, "is_repeated": false}}

        3. Query: "extract-based questions on Lencho"
        CORRECT: {{"subject": "english", "chapter": "a letter to god", "question_type": "extract-based"}}

        4. Query: "important questions from The Thief's Story"
        CORRECT: {{"subject": "english", "chapter": "the thief's story", "is_repeated": true}}

        ==== YOUR TASK ====

        Generate ONLY a JSON object for this query: {query}

        Check your JSON before submitting:
        - Remove any key with null, false, or empty values
        - Include only keys with meaningful information from the query
        - Format as clean JSON with no extra text
        """

        

        response = llm.invoke(system_prompt)

        response.content = response.content.lower()

        try:
            response_dict = json.loads(response.content)
            # Remove any keys with null, empty array, or false values
            response_dict = {k: v for k, v in response_dict.items() if v is not None and v != [] and v != False}
            print("Cleaned filter:", response_dict)
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            response_dict = {}

        print(response.content, type(response.content))

        response_dict = json.loads(response.content)
        filter_conditions = {}  # Initialize as empty dict

        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key=key,
                    match=MatchAny(any=value) if isinstance(value, list)
                        else MatchValue(value=value.lower() if isinstance(value, str) else value)
                )
                for key, value in response_dict.items()
            ]
        )



        print("Final filter:", filter_conditions)

        query_vector = embeddings.embed_query(query)  # Your query text embedded into a vector

        results = client.query_points(
            collection_name="questions",
            query=query_vector,
            with_payload=True,  
            query_filter=filter_conditions
        ).points


        print(results)
        print("processing output...")
        output = []

        iter = 1
        for i in results:
            
            print (f"Question {iter}: ", end = " ")
            print(i.payload["text"])
            print("")
            iter += 1

        
    elif ((intent.content) == "**advisory**" or (intent.content) == "`advisory`")or((intent.content) == "**both**" or (intent.content) == "`both`"):
        print("Sorry! For now, you can only ask direct questions based PYQs. Processing on general question in WIP. :( ")
        # with open("ai_project\Subjects\english\papers\extracted_questions.json", "r", encoding="utf-8") as f:
        #     context_data = json.load(f)

        # advise_prompt = ChatPromptTemplate.from_messages([
        #     ("system","you are a CBSE class X english literature teacher with the PYQs of the board paper. You are tasked to answe the doubt of the student, by reffering to this file: {json}. Only answer using the text file I gave you."),
        #     ("human","{input}"),
        # ])    




        # final_prompt = advise_prompt.format_messages(input = query, json = context_data)
        # print(final_prompt)

        # answer_advice = llm.invoke(final_prompt)
        # print(answer_advice.content)

    #------------------------------------------------------------------#



