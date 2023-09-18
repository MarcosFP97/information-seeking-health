import argparse
import re
import xml.etree.ElementTree as ET
from llama_cpp import Llama

MODEL = Llama(model_path="./models/llama-2-13b-chat.Q8_0.gguf", n_ctx=2000, n_gpu_layers=0)

def load_context(
  context:str
)-> str:
  if context:
    with open('../contexts/'+context+'.txt') as f:
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
  root = ET.parse('../evaluation/misinfo-resources-'+str(year)+'/topics/misinfo-'+str(year)+'-topics.xml').getroot()
  for topic in root.findall('topic'):
    if year=="2022":
      query = topic.find("question").text 
      answer = topic.find("answer").text 
    elif year=="2021":
      query = topic.find("description").text 
      answer = topic.find("stance").text
      if answer=="helpful":
        answer = 'yes'
      else:
        answer = 'no'
    elif year=="2020":
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
    outputfile = expert + str(year) + 'm.txt'
  else:
    outputfile = expert + str(year) + '.txt'

  with open('../outputs/zero-shot/llama/'+outputfile, 'w+') as f:
    for k, v in eval.items():
      if not must:
        prompt = get_prompt(context+' '+k)
      else:
        prompt = get_prompt(context+' '+k+' The answer must be yes or no.')
      print(prompt)
      f.write(prompt+'\n')
      output = MODEL(prompt, temperature=0, echo=False)
      response = output['choices'][0]['text']
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