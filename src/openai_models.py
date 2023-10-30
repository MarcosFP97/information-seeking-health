import re
import openai
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import tiktoken

API_KEY="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"

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
        f'<s>[INST] <<SYS>>\n'
        f'<</SYS>>\n'
        f'You are a helpful medical assistant.\n'
        f'{context}\n'
        f'Q: {question} A: [\INST]\n'
    )
  else:
    prompt = (
        f'{context}\n'
        f'Q: Will wearing an ankle brace help heal achilles tendonitis?\n'
        f'A: No.\n'
        f'Q: Does yoga improve the management of asthma?\n'
        f'A: Yes.\n'
        f'Q: Is starving a fever effective?\n'
        f'A: No.\n'
        f'Q: {question} A:\n'
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
    outputfile = expert + str(year) + '_3.txt'

  with open('../outputs/few-shot/'+model+'/'+outputfile, 'w+') as f:
    hits = 0
    for k, v in eval.items():
        prompt = get_prompt(context, syst, k)
        print(prompt)
        f.write(prompt+'\n')

        enc = tiktoken.encoding_for_model(model)
        logit_bias = {enc.encode("yes")[0]:50, enc.encode("no")[0]:50}

        if model=="gpt-3.5-turbo" or model=="gpt-4":
          response = openai.ChatCompletion.create(
              model=model,
              messages=[
                  {"role": "system", "content": "You are a chatbot"},
                  {"role": "user", "content": prompt},
              ],
              logit_bias=logit_bias,
              max_tokens=1,
              temperature=0 # to ensure reproducibility
          )

          result = ''
          for choice in response.choices:
              result += choice.message.content    

        else:
          response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=0,
            max_tokens=1,
            logit_bias=logit_bias,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
          )
          response = response['choices'][0]['text']+'\n'
          result = response.lower()

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
    openai.api_key = API_KEY
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs='?', default="gpt-4")
    parser.add_argument("context", nargs='?', default="expert") 
    parser.add_argument("year", nargs='?', default=2022)
    args = parser.parse_args()
    context = load_context(args.context)
    eval = load_answers(args.year)
    print(ask(args.model, eval, context, args.context, args.year))
