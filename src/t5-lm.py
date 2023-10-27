import argparse
import re
import xml.etree.ElementTree as ET
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-xl")


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
        f'{question}\n'
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
  
  n = len(eval)
  hits = 0

  if syst:
    outputfile = expert + str(year) + '_s.txt'
  else:
    outputfile = expert + str(year) + '.txt'

  with open('../outputs/zero-shot/flan-t5/'+outputfile, 'w+') as f:
    for k, v in eval.items():
      prompt = get_prompt(context, syst, k)
      print(prompt)
      f.write(prompt+'\n')
      inputs = TOKENIZER(prompt, return_tensors="pt") #.to('cuda:0')
      outputs = MODEL.generate(**inputs)
      response = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
      response = response[0].lower()
      print(response)
      f.write(response+'\n')

      m = re.search(r'\byes\b|\bno\b', response)
      if m and m.group().strip()==v:
        hits+=1
        f.write(str(hits)+'\n')
        print(hits)

    accuracy =  hits/n
    f.write("Accuracy:"+ str(accuracy))
    return accuracy


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("context", nargs='?', default="expert")
    parser.add_argument("year", nargs='?', default="2020")
    parser.add_argument("syst", nargs='?', default=True)
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(predict(context, eval, args.context, args.year, args.syst))