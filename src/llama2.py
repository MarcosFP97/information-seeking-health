import argparse
import re
import xml.etree.ElementTree as ET
import torch
from langchain import HuggingFacePipeline, PromptTemplate,  LLMChain
from transformers import AutoTokenizer

hf_auth = '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'
model = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token=hf_auth,
    device_map="auto",
    max_length=1000,
    eos_token_id=tokenizer.eos_token_id
)
MODEL = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})

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
    template = f"""
                {prompt} 
                {query} """
    prompt = PromptTemplate(template=template, input_variables=["prompt", "query"])
    llm_chain = LLMChain(prompt=prompt, llm=MODEL)
    response = llm_chain.run(prompt,k)    
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