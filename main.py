import os
# import openai
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings

from prompts import (
    router_promp,
    question_prompt,
    application_prompt,
    system_role_description,
    wellcome_text
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MAX_NEW_TOKENS,
    CHROMA_SETTINGS,
    DEVICE_TYPE
)


# Load the API key from the environment variable
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Langchain with OpenAI's GPT model
llm = OpenAIChat()

chat_model = ChatOpenAI()

embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})

vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )


chat_log = []
chat_log_user = []

client = OpenAI()

# def retrieve_information(query, search_type='similarity'):
#     # Encode the query to a vector and search in the vector database
#     query_vector = vector_db.search(query, search_type=search_type)
#     relevant_info = vector_db.search(query_vector, search_type=search_type)
#     return relevant_info

def retrieve_information(query):
    # Encode the query to a vector and search in the vector database
    query_vector = vector_db.search(query, search_type='similarity')
    # relevant_info = vector_db.search(query_vector, search_type='similarity')
    return str(query_vector)

def sent_to_gpt(chat_log):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    temperature=0.2,
    messages=chat_log)
    assistant_response = response.choices[0].message.content
    return assistant_response



chat_log.append({"role": "system", "content": system_role_description})



def chat_content(user_message):
    chat_log.append({"role": "user", "content": router_promp + user_message})
    router_response = sent_to_gpt(chat_log)
    relevant_data_info = retrieve_information(user_message)
    if router_response == "inquiry":
            prompt_with_details = question_prompt(user_message, relevant_data_info)
    elif router_response == "sign-up":
            prompt_with_details = application_prompt(user_message, relevant_data_info)      
    chat_log_user.append({"role": "user", "content": prompt_with_details})
    response_question = sent_to_gpt(chat_log_user)
    print("Summer Camp Assistant: \n", response_question.strip("\n").strip() + "\n") 
    
print(wellcome_text) 
    
    
while True:
  try:    
    print("Please enter your question: \n")
    user_message = input()
    if user_message.lower() == "quit":
        break
    else:
       chat_content(user_message)
    
  except Exception as e:
      print(f"An error occurred: {e}")