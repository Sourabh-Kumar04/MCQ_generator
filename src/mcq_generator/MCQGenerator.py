import os
import json
import requests
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcq_generator.utils import read_file, get_table_data
from src.mcq_generator.logger import  logging

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


# import necessay packages from langchain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# Load envirinment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm=HuggingFaceHub

# model_id="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model_id="meta-llama/Llama-3.2-3B-Instruct"
# model_id="google/flan-t5-large"
# model_id="meta-llama/Meta-Llama-3-8B-Instruct"

llm=llm(repo_id=model_id, model_kwargs={'temperature':0.1})

template="""
Text{text}
You are a expert MCQ maker. Given the above test, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the question are not repeated and checks all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below adn use it as guide. \
Ensure to make {number} MCQS
*** RESPONSE_JSON ***
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=['text', 'number', 'subject', "tone", 'response_json'],
    template=template
)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key='quiz', verbose=True)

template2="""
You are an expert english gramarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at max 50 words for complexity if the quiz is not at per with the congitive and analytical abilities of the students.\
Update the quiz questions which needs to be changed the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English writer of the above quiz:
"""

quiz_evaluation_prompt=PromptTemplate(
    input_variables=["subject", "quiz"],
    template=template2
)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key='review', verbose=True)

# This is the oberall chain which runs the two chains in Sequence
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)
