import argparse
import re
import xml.etree.ElementTree as ET
from transformers import AutoTokenizer
import transformers
import torch

MODEL = "meta-llama/Llama-2-7b-chat-hf"
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
        f'<s>[INST] <<SYS>>\n'
        f'<</SYS>>\n'
        f'You are a helpful medical assistant.\n'
        f'{context}\n'
        f'Q: {question} A: [\INST]\n'
    )
  else:
    prompt = (
        f'<s>[INST] <<SYS>>\n'
        f'<</SYS>>\n'
        f'{context}\n'
        f'Q: {question} A: [\INST]\n'
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
  syst:bool
)-> int:
  number_questions = len(eval)
  hits = 0
  
  if syst:
    outputfile = expert + str(year) + '_s.txt'
  else:
    outputfile = expert + str(year) + '.txt'

  with open('../outputs/zero-shot/llama/'+outputfile, 'w+') as f:
    for k, v in eval.items():
      prompt = get_prompt(context, syst, k)
      print(prompt)
      f.write(prompt+'\n')
      sequences = pipeline(
            prompt,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            max_length=2048,
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
    parser.add_argument("year", nargs='?', default="2021")
    parser.add_argument("syst", nargs='?', default=True)
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(predict(context, eval, args.context, args.year, args.syst))