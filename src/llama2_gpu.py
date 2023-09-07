import argparse
import re
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
import transformers
import torch

MODEL = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
pipeline = transformers.pipeline(
    "text-generation",
    model=MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)

def load_context(
  context:str
)-> str:
  if context:
    with open('./contexts/'+context) as f:
      context = f.read()
  return context

def get_prompt(
    message: str,
) -> str:
    prompt = (
        '<s>[INST] <<SYS>>\n'
        '\n<</SYS>>\n\n'
        f'"{message}" '
        f'This is your answer: [/INST]'
    )
    return prompt

def load_answers(
  year:int
)-> dict:
  answers = {}
  root = ET.parse('evaluation/misinfo-resources-'+str(year)+'/topics/misinfo-'+str(year)+'-topics.xml').getroot()
  for topic in root.findall('topic'):
    if year==2022:
      query = topic.find("question").text 
      answer = topic.find("answer").text 
    elif year==2021:
      query = topic.find("query").text 
      answer = topic.find("stance").text
    elif year==2020:
      query = topic.find("description").text 
      answer = topic.find("answer").text
    answers[query.rstrip()] = answer
  return answers

def predict(
  context:str,
  eval: dict,
  expert: str, 
  year: int,
  must:bool
)-> int:
  number_questions = len(eval)
  hits = 0
  
  if must:
    outputfile = expert + str(year) + '.txt'
  else:
    outputfile = expert + str(year) + 'm.txt'

  with open('./outputs/llama/'+outputfile, 'w+') as f:
    for k, v in eval.items():
      if not must:
        prompt = get_prompt(context+' '+k)
      else:
        prompt = get_prompt(context+' '+k+' The answer must be yes or no.')
      print(prompt)
      f.write(prompt+'\n')
      sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
      )
      response = ''
      for seq in sequences:
        response += seq['generated_text']
      response = response.lower()
      f.write(response+'\n')
      print(response)

      m = re.search(r'\byes\b|\bno\b', response)
      if m and m.group().strip()==v:
        hits+=1
        f.write(str(hits)+'\n')
        print(hits)
    accuracy =  hits/number_questions
    f.write("Accuracy:"+ str(accuracy))
    return accuracy

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("context", nargs='?', default="")
    parser.add_argument("year", nargs='?', default=2022)
    parser.add_argument("force", nargs='?', default=False)
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(predict(context, eval, args.context, args.year, args.force))