import argparse
import re
import xml.etree.ElementTree as ET
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
TOKENIZER = AutoTokenizer.from_pretrained("google/flan-t5-xl")

def load_prompt(path:str):
  with open(path) as f:
    content = f.read()
  return content

def load_answers(path:str):
  answers = dict()
  root = ET.parse(path).getroot()
  for topic in root.findall('topic'):
    query = topic.find("question").text 
    answer = topic.find("answer").text 
    answers[query.rstrip()] = answer
  return answers

def predict(prompt:str, answers: dict):
  n = len(answers)
  hits = 0
  for k, v in answers.items():
    t5_prompt = prompt + '\nQ:Will wearing an ankle brace help heal achilles tendonitis?\nA:No\nQ:Does yoga improve the management of asthma?\nA:Yes\nQ:Is starving a fever effective?\nA:No\n' + k + ' The answer must be Yes or No'
    print(t5_prompt)
    inputs = TOKENIZER(t5_prompt, return_tensors="pt")
    outputs = MODEL.generate(**inputs)
    response = TOKENIZER.batch_decode(outputs, skip_special_tokens=True)
    response = response.lower()
    print(response)

    m = re.search(r'\byes\b|\bno\b', response)
    if m and m.group().strip()==v:
      hits+=1
      print(hits)

  print("Accuracy:", hits/n)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs='?', default="prompts/h2oloo.txt")
    parser.add_argument("topics", nargs='?', default="evaluation/misinfo-resources-2022/topics/misinfo-2022-topics.xml")
    args = parser.parse_args()
    prompt = load_prompt(args.prompt)
    answers = load_answers(args.answers)
    predict(prompt, answers)