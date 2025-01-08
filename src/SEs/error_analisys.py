import sys
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

def calculate_precision_k(df, max_k):
    results = {}
    grouped = df.groupby('query')  # Agrupar por query

    for query, group in grouped:
        group = group.reset_index(drop=True)  # Reiniciar índices
        precisions = []

        for k in range(1, len(group) + 1):
            top_k = group.head(k)  # Seleccionar los primeros k resultados
            relevant_count = (top_k['hits'] == 1).sum()  # Contar relevantes
            precision = relevant_count / k  # Precisión = relevantes / k
            precisions.append(precision)

        results[query] = precisions

    return results

def individual(se:str, 
               year:int,
               question:str) \
               -> None:
    
    root = ET.parse("../llm/evaluation/misinfo-resources-"+str(year)+"/topics/misinfo-"+str(year)+"-topics.xml").getroot() ## open queries file
    query_ans = {}
    
    ####### ADAPTAR ESTO A VARIOS AÑOS
    for topic in root.findall('topic'):
        query=""
        if year==2021:
            query = topic.find("description").text
            answer = topic.find("stance").text
            if answer=="helpful":
                answer = 'yes'
            else:
                answer = 'no'
        else:
            if year==2022:
                query = topic.find("question").text
            elif year==2020:
                query = topic.find("description").text
            answer = topic.find("answer").text
        query_ans[query.rstrip()] = answer
    
    #print(query_ans)
    results = pd.read_csv("results/stance_prediction_"+se.lower()+"_"+str(year)+".csv", names=["query", "link", "passage", "answer", "label1", "label2"])
    results["label2"] = results["label2"].map(lambda x: str(x).replace("\n","").replace("\r","").replace(".", "").replace("no answer", "neutral"))
    results = results.dropna()
    #print(results.head())
    #grouped = results.groupby(results["query"])
    
    results["hits"] = [0]*len(results)
    results["tp"] = [0]*len(results)
    results["fp"] = [0]*len(results)
    results["tn"] = [0]*len(results)
    results["fn"] = [0]*len(results)
    for i in range(len(results)):
        query = results.iloc[i,:]["query"]
        #print(query)
        truth = query_ans[query]
        if truth=="no" and  results.iloc[i,:]["label2"]=="no": ### TRUE NEGATIVE
            results.loc[i,"hits"] = 1
            results.loc[i,"tn"] = 1
        elif truth=="no" and  results.iloc[i,:]["label2"]=="yes": #### FALSE POSITIVE
            results.loc[i,"hits"] = 0
            results.loc[i,"fp"] = 1
        elif truth=="yes" and  results.iloc[i,:]["label2"]=="no": #### FALSE NEGATIVE
            results.loc[i,"hits"] = 0
            results.loc[i,"fn"] = 1
        elif truth=="yes" and  results.iloc[i,:]["label2"]=="yes": #### TRUE POSITIVE
            results.loc[i,"hits"] = 1
            results.loc[i,"tp"] = 1
        else:
            results.loc[i,"hits"] = -1

    #print(results)
    #results.to_csv("results/stance_prediction_"+se+"_"+str(year)+"_hits.csv", header=True, index=False)
    precision_results = calculate_precision_k(results,20)
    #print(precision_results)
    return precision_results[question][9]

if __name__ == "__main__":
    year = int(sys.argv[1])

    if year==2020: ### This are the queries identified as difficult for the LLMs
        queries = ['Can bleach prevent COVID-19?', 'Can Nicotine help COVID-19?']
    elif year==2021:
        queries = ['Can fermented milk help mitigate high blood pressure?', 'Can evening primrose oil help treat eczema?', 'Can omega-3 treat borderline personality disorder in women?', 'Do ankle braces help heal an ankle fracture?']
    elif year==2022:
        queries = ['Can fish oil improve your cholesterol?', 'Can a speech problem be caused by right handed person being forced to left hand?']
    avg_ses = {}
    for query in queries:
        if not query in avg_ses:
            avg_ses[query] = []

        avg_ses[query].append(individual("Google", year, query))
        avg_ses[query].append(individual("Duckduckgo", year, query))
        avg_ses[query].append(individual("Yahoo", year, query))
        avg_ses[query].append(individual("Bing", year, query))
    
    query_difficulties = []
    for k,v in avg_ses.items():
        mean = np.mean(v)
        query_difficulties.append(mean)
    print("Difficult queries in",year,"have an average p@10 of",np.mean(query_difficulties))