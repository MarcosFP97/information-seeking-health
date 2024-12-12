import re
import openai
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import tiktoken
from ollama import chat
from ollama import ChatResponse

API_KEY=""

'''
This method loads a context from file.
Args:
    - path: path to the file containing the context
Return:
    - returns the context
'''
def load_context(
  context:str
)-> str:
  if context:
    with open('../contexts/'+context+'.txt') as f:
      context = f.read()
  return context


'''
This method generates a formatted prompt to input the model
Args:
    - message: main message to be included in the prompt
Return:
    - returns the formatted prompt
'''
def get_prompt(
    context: str,
    system:bool,
    question:str
) -> str:
  if system:
    prompt = (
        f'You are a helpful medical assistant.\n'
        f'{context}\n'
        f'Q: {question} A: \n'
    )
  else:
    prompt = (
        f'{context}\n'
        # f'Q: Will wearing an ankle brace help heal achilles tendonitis?\n'
        # f'A: No.\n'
        # f'Q: Does yoga improve the management of asthma?\n'
        # f'A: Yes.\n'
        # f'Q: Is starving a fever effective?\n'
        # f'A: No.\n'
        f'{question} \n'
    )

  return prompt


'''
This method loads the ground truth values for health questions from file.
Args:
    - path: path to the file containing the ground truth questions
Return:
    - returns a dictionary with the questions as keys and the answers as values
'''
def load_answers(
  year:int
)-> dict:
  answers = {}
  root = ET.parse('../evaluation/misinfo-resources-'+str(year)+'/topics/misinfo-'+str(year)+'-topics.xml').getroot()
  for topic in root.findall('topic'):
    if str(year)=="2022":
      query = topic.find("question").text 
      answer = topic.find("answer").text 
    elif str(year)=="2021":
      query = topic.find("description").text 
      answer = topic.find("stance").text
      if answer=="helpful":
        answer = 'yes'
      else:
        answer = 'no'
    elif str(year)=="2020":
      query = topic.find("description").text 
      answer = topic.find("answer").text
    answers[query.rstrip()] = answer
  return answers


'''
This method evaluates model's answer against ground truth values
Args:
    - response: model's answer to the question
    - answer: ground truth value
Return:
    - returns 1 if there is a math and 0 otherwise
'''
def evaluate(
    response:str,
    answer:str
)-> int:
    hit=0
    response = response.lower()

    m = re.search(r'\byes\b|\bno\b', response)
    if m and m.group().strip()==answer:
        hit=1
    return hit

'''
This method asks a LLM for the answer to a suite of health-related questions
Args:
    - model: model_name
    - prompt: context
    - answers: dict containing the questions plus their answers
Return:
    - returns the accuracy value for all the answers
'''
def ask(
  model:str,
  eval:dict,
  context:str,
  expert: str, 
  year: int,
  syst:bool=False,
)-> float:
  number_questions = len(eval)
  
  if syst:
    outputfile = expert + str(year) + '_s.txt'
  else:
    outputfile = expert + str(year) + '.txt'

  with open('../outputs/zero-shot/medllama3/'+outputfile, 'w+') as f:
    hits = 0
    for k, v in eval.items():
        prompt = get_prompt(context, syst, k)
        print(prompt)
        f.write(prompt+'\n')

        response: ChatResponse = chat(model=model, messages=[
          {
            'role': 'user',
            'content': prompt + '\nYou are only allowed to respond with either \'yes\' or \'no\'. Do not provide any other text or explanation.',
          },
        ])

        result = response['message']['content']

        f.write(result+'\n')
        print(result)
        hit = evaluate(result,v)
        if hit!=0:
            hits+=hit
            f.write(str(hits)+'\n')
            print(hits)

    accuracy =  hits/number_questions
    f.write("Accuracy:"+ str(accuracy))
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs='?', default="thewindmom/llama3-med42-8b") #  llama3.1:8b-instruct-q8_0
    parser.add_argument("context", nargs='?', default="") 
    parser.add_argument("year", nargs='?', default=2020)
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(ask(args.model, eval, context, args.context, args.year))
