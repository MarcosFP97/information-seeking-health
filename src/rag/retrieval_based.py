import re
import openai
import sys
import tiktoken
import pandas as pd
import argparse
from transformers import LlamaTokenizerFast
from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as ET
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
    with open('contexts/'+context+'.txt') as f:
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
    context:str,
    evidence:str,
    message: str,
) -> str:
    evidence = "".join([s for s in evidence.strip().splitlines(True) if s.strip("\r\n").strip()])
    prompt = (
        f'{context}\n'
        f'Provide an answer to the question using the provided evidence and contrasting it with your internal knowledge.\n'
        f'Evidence:"{evidence}"\n'
        f'Question:{message}\n'
        #f'Your answer:\n' # 
    )
    return prompt


def load_evidence(
    top:int,
    year:int
)-> dict:
    evidences = {}
    df = pd.read_csv('../SEs/results/stance_prediction_google_'+str(year)+'.csv', names=["query", "link", "passage", "answer", "label1", "label2"])
    
    if int(top)==123: ### merging the top3
      for _,g in df.groupby("query"):
        query = g.iloc[0,:]["query"] ### it is the same query
        top0 = g.iloc[0,:]["passage"]
        top1 =  g.iloc[1,:]["passage"]
        top2 = g.iloc[2,:]["passage"]
        top = top0 + '\n' + top1 + '\n' + top2
        evidences[query] = top

    else:
      for _,g in df.groupby("query"):
          evidences[g.iloc[int(top)-1,:]["query"]] = g.iloc[int(top)-1,:]["passage"]
    
    return evidences

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
  root = ET.parse('evaluation/misinfo-resources-'+str(year)+'/topics/misinfo-'+str(year)+'-topics.xml').getroot()
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
  context:str,
  evidences:dict,
  eval: dict,
  expert: str, 
  year: int,
  top:int
)-> float:
  number_questions = len(eval)
  
  outputfile = expert + str(year) + '_' + str(top)+'.txt'
  path = 'outputs/'+model+'/'+outputfile
  if "llama" in model:
    path = 'outputs/llama3/'+outputfile
  with open(path, 'w+') as f:
    hits = 0
    for k, v in eval.items():

        passage = evidences[k] ### The passage without any truncation
        # max_length=0
        # if "llama" in model: #### TRUNCATING THE PASSAGE
        #   max_length=2048 - 109 #### (VAR only used for LLAMA2 and D-002) the real max size is 2048 but we need to also take into account the tokens of the "common" parts + the question (we tried to mantain room for longer questions)
        #   if "expert" in context:
        #     max_length = 2048 - 165 ### is less because we need to save tokens for the context part
        #   logit_bias = {1217:50, 3582:50} ### codificación para Llama
        #   tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer", model_max_length=max_length)
        #   if len(tokenizer.encode(passage)) > max_length: 
        #     tokens = tokenizer.encode(passage, truncation=True)
        #     passage = tokenizer.decode(tokens)
        # else: 
        #   enc = tiktoken.encoding_for_model(model) ### codificación para los diferentes modelos de OpeAI
        #   logit_bias = {enc.encode("yes")[0]:50, enc.encode("no")[0]:50}
        #   if "text-davinci-002" in model or "gpt-3.5-turbo" in model:
        #     max_length= 4096 - 109
        #     if "expert" in context:
        #       max_length = 4096 - 165 ### is less because we need to save tokens for the context part
        #     if len(enc.encode(passage)) > max_length: #### the real max size is 2048 but we need to also take into account the tokens of the "common" parts + the question (we tried to mantain room for longer questions)
        #       tokens = enc.encode(passage)
        #       passage = enc.decode(tokens[:max_length])

        prompt = get_prompt(context, passage, k)
        print(prompt)
        f.write(prompt+'\n')
        
        response: ChatResponse = chat(model=model, messages=[
          {
            'role': 'user',
            'content': prompt + '\nYou are only allowed to respond with either \'yes\' or \'no\'. Do not provide any other text or explanation.\nYour Answer:',
          },
          
        ],options={'temperature': 0})

        result = response['message']['content']

        # if "llama" in model or model=="gpt-3.5-turbo" or model=="gpt-4":
        #   try:
        #     response = openai.ChatCompletion.create(
        #       model=model,
        #       messages=[
        #           {"role": "system", "content": "You are a chatbot"},
        #           {"role": "user", "content": prompt},
        #       ],
        #       logit_bias=logit_bias, 
        #       max_tokens=1,
        #       temperature=0 # to ensure reproducibility
        #     )

        #     result = ''
        #     for choice in response.choices:
        #       result += choice.message.content    
        #   except:
        #       result=f'!!!!!ERROR:{e}'

        # else:
        #   try:
        #     response = openai.Completion.create(
        #       engine=model,
        #       prompt=prompt,
        #       temperature=0,
        #       max_tokens=1,
        #       logit_bias=logit_bias,
        #       top_p=1.0,
        #       frequency_penalty=0.0,
        #       presence_penalty=0.0
        #     )
        #     response = response['choices'][0]['text']+'\n'
        #     result = response.lower()
        #   except Exception as e:
        #        result=f'!!!!!ERROR:{e}'
              
        f.write(result+'\n')
        print(result)
        hit = evaluate(result,v)
        if hit!=0:
            hits+=hit
            f.write(str(hits)+'\n\n')
            print(hits)
       

    accuracy =  hits/number_questions
    f.write("Accuracy:"+ str(accuracy))
    return accuracy

if __name__ == "__main__":
    openai.api_key = API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', nargs='?', default="llama3.1:8b-instruct-q8_0") # thewindmom/llama3-med42-8b 
    parser.add_argument("--context", '-c', nargs='?', default="") # the other context we are going to evaluate with retrieval-based approach is expert
    parser.add_argument("--top", '-t', nargs='?', default=1)
    parser.add_argument("--year", '-y', nargs='?', default=2020)
    args = parser.parse_args()

    if "llama" in args.model: ### we change the endpoint for Llama (it needs that the server is listening)
      openai.api_base = "http://localhost:8000/v1"
    
    context = load_context(args.context)
    evidences = load_evidence(args.top, args.year)
    eval = load_answers(args.year)
    print(ask(args.model, context, evidences, eval,  args.context, args.year, args.top))
