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
    
    for f in os.listdir('./pickle_hits/'+year+'/'):
        model = f.split('_')
        with open('./pickle_hits/'+year+'/'+f, 'rb') as f:
            hits = pickle.load(f)
            aux[model[0]] = hits

    gpt4 = aux.groupby("Perc_Corr")["gpt-4"].mean().reset_index()
    print(gpt4)
    llama3 = aux.groupby("Perc_Corr")["llama3"].mean().reset_index()
    print(llama3)
    medllama3 = aux.groupby("Perc_Corr")["medllama3"].mean().reset_index()
    print(medllama3)
    gpt35 = aux.groupby("Perc_Corr")["gpt-3.5-turbo"].mean().reset_index()
    print(gpt35)
    d002 = aux.groupby("Perc_Corr")["text-davinci-002"].mean().reset_index()
    print(d002)
    x_ticks = [0/3, 1/3, 2/3, 3/3]
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, gpt4['gpt-4'].values, marker='o', label='GPT-4')
    plt.plot(x_ticks, llama3['llama3'].values, marker='x', label='Llama3')
    plt.plot(x_ticks, medllama3['medllama3'].values, marker='^', label='MedLlama3')
    plt.plot(x_ticks, gpt35['gpt-3.5-turbo'].values, marker='s', label='ChatGPT')
    plt.plot(x_ticks, d002['text-davinci-002'].values, linewidth=1.5, marker='d', label='text-davinci-002')

    plt.xlabel('Proportion of correct passages in RAG', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    # plt.title(f'Model Performance for {year}')
    plt.xticks(x_ticks, labels=['0/3', '1/3', '2/3', '3/3'], fontsize=16)
    plt.ylim(0, 1.05)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig('./figs/' + year + '.png', bbox_inches='tight')
    plt.show()