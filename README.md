# Information-Seeking in Health with LLMs, Search Engines and RAG approaches

## Overview
This repository evaluates four well-known search engines, seven Large Language Models (LLMs) and "online" RAG experiments for health information seeking. The evaluation leverages TREC Health Misinformation datasets as benchmarks.

## Citation
We have already published a conference paper. Please **cite us** if you use this repo.

```bibtex
@inproceedings{fernandez2024binaryQA,
  title={Large Language Models for Binary Health-Related Question Answering: A Zero- and Few-Shot Evaluation},
  author={Fernández-Pichel, Marcos and Losada, David E. and Pichel, José C.},
  booktitle={Computational Science -- ICCS 2024},
  year={2024},
  publisher={Springer, Cham},
  volume={14835},
  pages={xxx-yyy},
  doi={10.1007/978-3-031-63772-8_29}
}
```
Currently, we have a journal paper **under review** with all the add-ons to the first publication.

### Key Features
- **Health-Related QA**: Focused on binary questions to address health misinformation.
- **Benchmarks**: Uses TREC HM datasets for rigorous evaluation.
- **Search engines**: We evaluated Google, Duckduckgo, Yahoo and Bing.
- **Prompts and Strategies**: Incorporates advanced prompting strategies for zero- and few-shot learning.
- **RAG**: We propose a novel "online" RAG approach injecting 
- **Continuous Updates**: Adapts to new LLMs. We have just included Llama3 and MedLlama3 into our evaluation.

## Contents
- `src/llm`: Contains scripts for experiments, data processing, and large language model evaluation.
- `src/mcnemar`: Contains a script to compute McNemar stats tests between LLM answer files.
- `src/rag`: Contains all the scripts to reproduce the RAG experiments listed in our work.
- `src/SEs`: Contains all the scripts to reproduce the Search Engines experiments listed in our work.

## Requirements
- Python 3.8+
- Jupyter Notebook

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/MarcosFP97/information-seeking-health.git
   cd information-seeking-health
   ```
3. Run the Jupyter notebooks and scripts in `src/` folder as needed.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributions
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request for review. 

## Contact
For questions or feedback, please contact marcosfernandez.pichel@usc.es.
