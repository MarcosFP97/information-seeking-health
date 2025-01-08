import re
import sys
from typing import List

def get_incorrect_queries(
    model: str,
    year:str,
    context:str
) -> List[str]:
    if "llama" in model: ### different pattern to process llama files
        pattern1 = re.compile(r'^[A-Za-z0-9\(\)\-\'\"\s\!\.\,]*?\?([A-Za-z0-9\(\)\-\'\"\s\:])*(\[\\INST\])?$')
        extension = '.txt'
    else:
        pattern1 = re.compile(r'[A-Za-z0-9\(\)\-\'\s]*\?')
        extension = '_verbose.txt'

    pattern2 = re.compile(r'^[0-9]+\n')
    stoppattern = re.compile(r'Accuracy:[0-9].[0-9]+\n')
    is_query = False

    qs = []
    with open("../outputs/zero-shot/"+model+'/'+context+year+extension, 'r') as pf:
        data = pf.readlines()
        
        query = ""    
        for line in data:
            is_q = pattern1.match(line)
            is_n = pattern2.findall(line)
            is_stop = stoppattern.match(line)
            if is_q:
                # print("Query:",is_q[0])
                if is_query:
                    if "llama": 
                        query = query.replace("\n","").rstrip()
                    qs.append(query)
                    # is_query = False
                    query = is_q[0]
                else:
                    query = is_q[0]
                    is_query = True
            elif is_query and is_n:
                # print("Números", is_n[0])
                is_query = False
                query = ""
            elif is_stop and is_query:
                # print("Stop:", is_stop[0])
                qs.append(query)
            else:
                continue

        return qs

if __name__=="__main__":
    params = sys.argv
    year = sys.argv[1]
    med_llama_qs = get_incorrect_queries("medllama3", year, "")
    llama_qs = get_incorrect_queries("llama3", year, "")
    gpt4_qs = get_incorrect_queries("gpt-4", year, "")
    gpt35_qs = get_incorrect_queries("gpt-3.5-turbo", year, "")
    print(f'COMMON INCORRECT QUERIES FOR NO CONTEXT:{set(med_llama_qs) & set(llama_qs) & set(gpt4_qs) & set(gpt35_qs)}')

    med_llama_qs = get_incorrect_queries("medllama3", year, "expert")
    llama_qs = get_incorrect_queries("llama3", year, "expert")
    gpt4_qs = get_incorrect_queries("gpt-4", year, "expert")
    gpt35_qs = get_incorrect_queries("gpt-3.5-turbo", year, "expert")
    print(f'COMMON INCORRECT QUERIES FOR EXPERT CONTEXT:{set(med_llama_qs) & set(llama_qs) & set(gpt4_qs) & set(gpt35_qs)}')