from mlxtend.evaluate import mcnemar, mcnemar_table
from tracemalloc import stop
import xml.etree.ElementTree as ET
import numpy as np
import argparse
import re
from typing import List

def get_labels(run: str) -> List[int]:
    hits = []

    if "llama" in run: ### different pattern to process llama files
        pattern1 = re.compile(r'^[A-Za-z0-9\(\)\-\'\"\s\!\.\,]*?\?([A-Za-z0-9\(\)\-\'\"\s\:])*(\[\\INST\])?$')
    else:
        pattern1 = re.compile(r'[A-Za-z0-9\(\)\-\'\s]*\?')

    # if "_" in run: ## means ICL
    #         pattern1 = re.compile(r'^Q: [A-Za-z0-9\(\)\-\'\"\s\!\.\,]*?\? A:(\[\\INST\])?$')

    pattern2 = re.compile(r'^[0-9]+\n')
    stoppattern = re.compile(r'Accuracy:[0-9].[0-9]+\n')
    query = False

    with open("../../outputs/"+run, 'r') as pf:
        data = pf.readlines()
        count = 0
        for line in data:
            #line = line.replace('\n','')
            is_q = pattern1.match(line)
            is_n = pattern2.findall(line)
            is_stop = stoppattern.match(line)
            # print(f'\"{line}\"')
            # print()
            # print()
            if is_stop and query:
                #print("Stop:", is_stop[0])
                hits.append(0)

            elif query and is_n:
                #print("Números", is_n[0])
                hits.append(1)
                query = False
            
            if is_q:
                #print("Query:",is_q[0])
                count+=1
                if query:
                    hits.append(0)
                else:
                    query = True
            else:
                continue
    # print("Hits",hits) #### CHEQUEAR POR QUÉ 0.88
    print(len(hits))
    print(np.mean(hits))
    return hits

def eval_mcnemar(y_target: List, y_model1: List, y_model2: List):
    tb = mcnemar_table(y_target=y_target, 
                   y_model1=y_model1, 
                   y_model2=y_model2)
    # print(tb)
    _, p = mcnemar(ary=tb, corrected=True)
    if p>0.05:
        print("We cannot reject the null hypothesis and there is not significant difference between classifiers", p)
    else:
        print("There is significant difference", p)
    print()
    print()

''' 
This code parses the two runs files, computes its accuracy and list of hits, and applies McNemar's test
'''
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year")
    parser.add_argument("run1")
    parser.add_argument("run2")
    args = parser.parse_args()
    root = ET.parse("../../evaluation/misinfo-resources-"+str(args.year)+"/topics/misinfo-"+str(args.year)+"-topics.xml").getroot()
    
    ground_truth = {}
    for topic in root.findall('topic'):
        if args.year=="2021":
            query = topic.find("description").text
            answer = topic.find("stance").text
            if answer=="helpful":
                answer = 'yes'
            else:
                answer = 'no'
        else:
            if args.year=="2022":
                query = topic.find("question").text
            elif args.year=="2020":
                query = topic.find("description").text
            answer = topic.find("answer").text

        if answer=="yes":
            ground_truth[query] = 1
        else:
            ground_truth[query] = 0

    # The correct target (class) labels
    y_target = np.array(list(ground_truth.values())) #### esto lo sacaría del fichero de xml de 2022 por ejemplo
    print(len(y_target))
    # # Class labels predicted by model 1
    model1= get_labels(args.run1)
    # print(len(model1))
    y_model1 = np.array(model1) ### esta sería una run de 2022 de un clasificador y abajo del mismo o de otro

    # Class labels predicted by model 2
    model2 = get_labels(args.run2)
    # print(len(model2))
    y_model2 = np.array(model2)
    eval_mcnemar(y_target, y_model1, y_model2)