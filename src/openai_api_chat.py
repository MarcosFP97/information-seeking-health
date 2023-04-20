import os
import re
import openai
import typer
import time
import argparse
import xml.etree.ElementTree as ET
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

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

def ask_gpt(model:str,
            prompt: str,
            answers: dict):
  n = len(answers)
  hits = 0
  for k, v in answers.items():
    gpt_prompt =  '\nQ:Will wearing an ankle brace help heal achilles tendonitis?\nA:No\nQ:Does yoga improve the management of asthma?\nA:Yes\nQ:Is starving a fever effective?\nA:No\n' + prompt +'\n'+ k + ' The answer must be Yes or No' 
    print(gpt_prompt)

    response = openai.ChatCompletion.create(
      model=model,
      messages=[
          {"role": "system", "content": "You are a chatbot"},
          {"role": "user", "content": gpt_prompt},
      ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    print(result)
    
    if model == "text-davinci-002": #### if the model is less obedient
      m = re.search(r'\bno definitive\b|\bno clear\b|\bno certain\b|\bno consensus\b', result.lower())
      if m:
        continue

    m = re.search(r'\byes\b|\bno\b', result.lower())
    if m and m.group().strip()==v:
      hits+=1
      print(hits)
    
    print()
    print()

  print("Accuracy:", hits/n)    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("model", nargs='?', default="gpt-4")
  parser.add_argument("prompt", nargs='?', default="prompts/usc.txt")
  parser.add_argument("topics", nargs='?', default="evaluation/misinfo-resources-2022/topics/misinfo-2022-topics.xml")
  args = parser.parse_args()
  openai.api_key = API_KEY
  prompt = load_prompt(args.prompt)
  answers = load_answers(args.topics)
  ask_gpt(args.model, prompt, answers)
  