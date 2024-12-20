import pandas as pd
import xml.etree.ElementTree as ET
from scipy.stats import mannwhitneyu
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

def get_precisions(
    se:str, 
    year:int
) -> None:
    
    root = ET.parse("../../../evaluation/misinfo-resources-"+str(year)+"/topics/misinfo-"+str(year)+"-topics.xml").getroot() ## open queries file
    query_ans = {}
    if year==2022:
        for topic in root.findall('topic'):
            query = topic.find("question").text
            answer = topic.find("answer").text
            query_ans[query.rstrip()] = answer
    elif year==2021:
        for topic in root.findall('topic'):
            query = topic.find("description").text
            answer = topic.find("stance").text
            query_ans[query.rstrip()] = answer
            if answer=="unhelpful": ##### ADAPTAR ESTO A VARIOS AÑOS
                query_ans[query.rstrip()] = "no"
            else:
                query_ans[query.rstrip()] = "yes"
    elif year==2020:
        for topic in root.findall('topic'):
            query = topic.find("description").text
            answer = topic.find("answer").text
            query_ans[query.rstrip()] = answer
    
    results = pd.read_csv("../results/stance_prediction_"+se.lower()+"_"+str(year)+".csv", names=["query", "link", "passage", "answer", "label1", "label2"])
    results["label2"] = results["label2"].map(lambda x: str(x).replace("\n","").replace("\r","").replace(".", "").replace("no answer", "neutral"))
    results = results.dropna()
    
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

    precision_results = calculate_precision_k(results,20)
    values_at_k = {}
    for k in range(20):
        values_at_k[k+1] = [precision_results[query][k] for query in precision_results if len(precision_results[query]) > k]
    return values_at_k

def test(
    se1:str,
    se2:str,
    year:int,
    pos:int
):
    pre1 = get_precisions(se1,year)
    pre2 = get_precisions(se2,year)
   
    if np.mean(pre1[pos]) > np.mean(pre2[pos]):
        _, p = mannwhitneyu(pre1[pos], pre2[pos],  alternative='greater')
        if p<.05:
            print(f'There is significant difference between {se1} ({np.mean(pre1[pos])}) and {se2} ({np.mean(pre2[pos])} for year {str(year)} and p={str(pos)}')
    
    if np.mean(pre1[pos]) < np.mean(pre2[pos]):
        _, p = mannwhitneyu(pre2[pos], pre1[pos],  alternative='greater')
        if p<.05:
            print(f'There is significant difference between {se1} ({np.mean(pre1[pos])}) and {se2} ({np.mean(pre2[pos])} for year {str(year)} and p={str(pos)}')

if __name__=="__main__":
    test("Bing", "Yahoo", 2020, 1)
    test("Bing", "Google", 2020, 1)
    test("Bing", "Duckduckgo", 2020, 1)
    test("Google", "Yahoo", 2020, 1)
    test("Google", "Duckduckgo", 2020, 1)
    test("Duckduckgo", "Yahoo", 2020, 1)

    test("Bing", "Yahoo", 2020, 20)
    test("Bing", "Google", 2020, 20)
    test("Bing", "Duckduckgo", 2020, 20)
    test("Google", "Yahoo", 2020, 20)
    test("Google", "Duckduckgo", 2020, 20)
    test("Duckduckgo", "Yahoo", 2020, 20)

    test("Bing", "Yahoo", 2021, 1)
    test("Bing", "Google", 2021, 1)
    test("Bing", "Duckduckgo", 2021, 1)
    test("Google", "Yahoo", 2021, 1)
    test("Google", "Duckduckgo", 2021, 1)
    test("Duckduckgo", "Yahoo", 2021, 1)

    test("Bing", "Yahoo", 2021, 20)
    test("Bing", "Google", 2021, 20)
    test("Bing", "Duckduckgo", 2021, 20)
    test("Google", "Yahoo", 2021, 20)
    test("Google", "Duckduckgo", 2021, 20)
    test("Duckduckgo", "Yahoo", 2021, 20)

    test("Bing", "Yahoo", 2022, 1)
    test("Bing", "Google", 2022, 1)
    test("Bing", "Duckduckgo", 2022, 1)
    test("Google", "Yahoo", 2022, 1)
    test("Google", "Duckduckgo", 2022, 1)
    test("Duckduckgo", "Yahoo", 2022, 1)

    test("Bing", "Yahoo", 2022, 20)
    test("Bing", "Google", 2022, 20)
    test("Bing", "Duckduckgo", 2022, 20)
    test("Google", "Yahoo", 2022, 20)
    test("Google", "Duckduckgo", 2022, 20)
    test("Duckduckgo", "Yahoo", 2022, 20)