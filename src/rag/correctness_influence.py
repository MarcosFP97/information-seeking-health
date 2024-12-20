import pandas as pd
import xml.etree.ElementTree as ET
import sys
import pickle
import matplotlib.pyplot as plt
import os

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

if __name__=="__main__":
    year = sys.argv[1]
    google = pd.read_csv(f'../SEs/results/stance_prediction_google_{str(year)}.csv', names=["query", "link", "passage", "answer", "label1", "answergpt"])
    google = google[['query', 'passage', 'answergpt']]
    answers = load_answers(year)
    grouped_google = google.groupby('query', sort=False)

    corr_pass = {}
    top = 3
    for _, g in grouped_google:
        corr_passgs = 0
        for index, row in g[:top].iterrows():
            query = row["query"]
            answergpt = row["answergpt"].replace('\n','').replace('.','')
            if answers[query]==answergpt:
                corr_passgs+=1
        corr_pass[query] = corr_passgs/top
    
    aux = pd.DataFrame(corr_pass.items(), columns=['Query', 'Perc_Corr'])
    
    for f in os.listdir('./pickle_hits/2020/'):
        model = f.split('_')
        with open('./pickle_hits/2020/'+f, 'rb') as f:
            hits = pickle.load(f)
            aux[model[0]] = hits

    gpt4 = aux.groupby("Perc_Corr")["gpt-4"].mean().reset_index()
    llama3 = aux.groupby("Perc_Corr")["llama3"].mean().reset_index()
    medllama3 = aux.groupby("Perc_Corr")["medllama3"].mean().reset_index()
    gpt35 = aux.groupby("Perc_Corr")["gpt-3.5-turbo"].mean().reset_index()
    d002 = aux.groupby("Perc_Corr")["text-davinci-002"].mean().reset_index()
    plt.figure(figsize=(8,5))
    plt.plot(gpt4['Perc_Corr'].values, gpt4['gpt-4'].values)
    plt.plot(llama3['Perc_Corr'].values, llama3['llama3'].values)
    plt.plot(medllama3['Perc_Corr'].values, medllama3['medllama3'].values)
    plt.plot(gpt35['Perc_Corr'].values, gpt35["gpt-3.5-turbo"].values)
    plt.plot(d002['Perc_Corr'].values, d002["text-davinci-002"].values)
    plt.legend()
    plt.show()