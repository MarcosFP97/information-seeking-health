import typer
import pandas as pd
import xml.etree.ElementTree as ET

app = typer.Typer()

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

@app.command()
def individual(se:str=typer.Argument("Duckduckgo"), 
               year:int=typer.Argument(2022)) \
               -> None:
    
    root = ET.parse("../../evaluation/misinfo-resources-"+str(year)+"/topics/misinfo-"+str(year)+"-topics.xml").getroot() ## open queries file
    query_ans = {}
    
    ####### ADAPTAR ESTO A VARIOS AÑOS
    for topic in root.findall('topic'):
        query = topic.find("question").text
        answer = topic.find("answer").text
        query_ans[query.rstrip()] = answer
        # if answer=="unhelpful": ##### ADAPTAR ESTO A VARIOS AÑOS
        #     query_ans[query.rstrip()] = "no"
        # else:
        #     query_ans[query.rstrip()] = "yes"
    
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
    # precision_df = pd.DataFrame.from_dict(precision_results, orient='index')
    # precision_df.columns = [f"P@{k}" for k in range(1, precision_df.shape[1] + 1)]
    # precision_df.index.name = "Query"
    # precision_df.reset_index(inplace=True)
    # print(precision_df)
    precision_averages = []
    num_queries_per_k = []  # Para registrar cuántas queries contribuyen a cada k

    for k in range(20):
        values_at_k = [precision_results[query][k] for query in precision_results if len(precision_results[query]) > k]
        # if values_at_k:
        precision_averages.append(sum(values_at_k) / len(values_at_k))
        num_queries_per_k.append(len(values_at_k))

    print(precision_averages)
    print(num_queries_per_k)
    final_df = pd.DataFrame({
    "top": range(1, 20 + 1),
    "Precision_Average": precision_averages,
    "se": [se] * 20,  # Cambia 42 por el valor constante que necesites
    "Num_Queries_Contributing": num_queries_per_k,
    })
    print(final_df)
    final_df.to_csv(f'./precision_results/precision_{se.lower()}_{year}.csv', index=False, header=True)

if __name__ == "__main__":
	app()