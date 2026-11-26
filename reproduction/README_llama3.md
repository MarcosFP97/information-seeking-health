# Llama 3.1 8B — Reproduction Study

## 1. Objective

This reproduction is conducted as part of the Open Science
course at TH Köln.

The objective is to reproduce a subset of the experiments
presented in:

Fernández-Pichel, M., Pichel, J. C., & Losada, D. E. (2025).
Evaluating search engines and large language models for
answering health questions.
npj Digital Medicine, 8, 153.

Original repository:
https://github.com/MarcosFP97/information-seeking-health

## 2. Experimental Scope

- Model: Llama 3.1 8B Instruct
- Dataset: TREC Health Misinformation 2022
- Number of questions: 50
- Task: Binary health-related question answering
- Learning strategy: Zero-shot
- Prompt configurations:
  - No-context
  - Non-expert
  - Expert
- Total question-prompt evaluations: 150

This reproduction does not cover the other models,
datasets, few-shot experiments, search engines or
RAG experiments presented in the original paper.

## 3. Experimental Environment

| Component | Specification |
|---|---|
| Experiment date | 2026-11-26 |
| Computer | Apple MacBook |
| Hardware identifier | Mac17,3 |
| Processor | Apple M5 |
| Unified memory | 16 GB |
| Operating system | macOS 27.0.1 |
| Python | 3.12.16 |
| Ollama | 0.40.18 |
| Ollama Python package | 0.6.3 |
| Model | llama3.1:8b-instruct-q8_0 |
| Local model identifier | b158ded76fa0 |
| Quantization | Q8_0 |
| Temperature | 0 |
| Context window | 8192 tokens |
| Few-shot examples | 0 |

The dependency snapshot is available in:

    reproduction/env_llama3.txt

The base repository commit used for the experiments was:

    4729dd4eb3556872db09a7396fa609d18b902436

Local code modifications were present during execution,
as recorded in the experimental JSON logs.

## 4. Environment Setup

Install Python 3.12 and Ollama before proceeding.

Clone the reproduction repository:

    git clone https://github.com/Msuhxn/information-seeking-health-reproduction.git
    cd information-seeking-health-reproduction

Create and activate a Python virtual environment:

    python3.12 -m venv .venv
    source .venv/bin/activate

Install the recorded dependencies:

    python -m pip install -r reproduction/env_llama3.txt
    python -m pip check

Download the model:

    ollama pull llama3.1:8b-instruct-q8_0

Verify model availability:

    ollama list

Ensure that the Ollama service is running before
executing the experiments.

Use the repository revision containing the reproduction
scripts and modifications described in this document.

## 5. Code Modifications

The following changes were made to support reliable
local execution and reproducibility:

- Corrected dataset and context file paths.
- Used script-relative paths instead of relying on
  the current working directory.
- Created a separate reproduction output directory.
- Added protection against overwriting prediction files.
- Added a dedicated experimental runner.
- Recorded hardware and software metadata.
- Recorded execution time and accuracy.
- Preserved the repository's prompt wording.
- Preserved the original answer evaluation logic.
- Recorded the Python dependency versions.

The original experimental output files were not
intentionally modified.

## 6. Experimental Procedure

A preliminary pilot experiment was conducted using
the first three questions to verify model execution,
prediction storage and evaluation.

The pilot was excluded from the final results.

The full experiments can be executed from the
repository root using:

    python reproduction/run_llama3.py no-context
    python reproduction/run_llama3.py non-expert
    python reproduction/run_llama3.py expert

Each experiment processes all 50 questions.

The runner creates a timestamped output directory
containing the predictions and a JSON metadata log.

## 7. Prompt Configurations

### No-context

The question is presented without an additional
persona or contextual description.

### Non-expert

We are a non-expert user searching for medical
advice online.

### Expert

We are a committee of leading scientific experts
and medical doctors reviewing the latest and highest
quality of research from PubMED. For each question,
we have chosen an answer, either 'yes' or 'no',
based on our best understanding of current medical
practice and literature.

The repository's original context files were used.

The non-expert wording in the repository differs
slightly from the wording reported in the paper.

All configurations use the repository's instruction
restricting model responses to yes or no.

## 8. Evaluation Method

Predictions are evaluated against the ground-truth
answers provided by the TREC 2022 dataset.

The original evaluation logic extracts the first
occurrence of yes or no from the model response.

Accuracy is calculated as:

    Accuracy = Correct predictions / Total questions

The stored predictions were independently evaluated
using the team's compare_answers.py script.

Correctness refers to agreement with benchmark
reference labels, not independent medical validation.

## 9. Experimental Results

| Prompt | Paper accuracy | Reproduction | Difference |
|---|---:|---:|---:|
| No-context | 82% | 88% | +6 pp |
| Non-expert | 80% | 82% | +2 pp |
| Expert | 80% | 84% | +4 pp |
| Mean | 80.67% | 84.67% | +4 pp |

pp = percentage points.

### Runtime

| Prompt | Total seconds | Seconds per question |
|---|---:|---:|
| No-context | 17.1264 | 0.3425 |
| Non-expert | 15.7368 | 0.3147 |
| Expert | 13.8547 | 0.2771 |

Total measured runtime: 46.718 seconds.

No runtime errors were recorded during the three
full experiments.

The measured runtime includes the inference and
evaluation procedure and prediction file writing.
Model loading and warm-up conditions were not
controlled across configurations.

## 10. Question-Level Comparison

| Metric | No-context | Non-expert | Expert |
|---|---:|---:|---:|
| Identical predictions | 47 | 49 | 46 |
| Different predictions | 3 | 1 | 4 |
| Prediction agreement | 94% | 98% | 92% |
| Improved predictions | 3 | 1 | 3 |
| Worsened predictions | 0 | 0 | 1 |

Overall, 142 of the 150 predictions matched
the authors' saved predictions, representing
94.67% agreement.

Seven predictions changed from incorrect to correct,
while one prediction changed from correct to incorrect.

Detailed comparisons are available in:

    reproduction/outputs/llama3/analysis/

## 11. Statistical Analysis

The exact two-sided McNemar test was used to compare
paired prediction correctness.

A significance level of 0.05 was selected.

| Prompt | Original only correct | Reproduction only correct | P-value |
|---|---:|---:|---:|
| No-context | 0 | 3 | 0.2500 |
| Non-expert | 0 | 1 | 1.0000 |
| Expert | 1 | 3 | 0.6250 |

None of the differences were statistically significant.

The statistical calculations can be reproduced using:

    python reproduction/mcnemar_llama3.py

This script reads the paired correctness counts from
summary.csv and verifies the recorded p-values.

The summary CSV was manually consolidated from
the experimental logs and question-level comparisons.

## 12. Output Artifacts

All reproduction outputs are stored in:

    reproduction/outputs/llama3/

The directory contains:

- pilot/: Preliminary experiment.
- no-context/: Full no-context experiment.
- non-expert/: Full non-expert experiment.
- expert/: Full expert experiment.
- analysis/: Comparison and statistical results.

Each full experiment contains its raw predictions
and experimental metadata.

The analysis directory contains:

- summary.csv
- compare_no_context.txt
- compare_non_expert.txt
- compare_expert.txt
- mcnemar_results.txt

## 13. Limitations

- Only one model and one dataset were reproduced.
- Only one full run per prompt configuration was
  conducted, limiting assessment of run-to-run stability.
- Exact numerical agreement with the original paper
  was not achieved.
- Differences in hardware, software versions and model
  execution may contribute to prediction differences,
  but their individual effects were not established.
- Temperature zero does not guarantee identical
  predictions across all execution environments.
- The dataset contains only 50 questions per
  configuration, limiting statistical power.
- Statistical non-significance does not establish
  equivalence between the experiments.
- Runtime comparisons against the original study
  were not performed.
- The same questions were reused across configurations;
  the 150 evaluations are not 150 independent questions.

## 14. Contributions to FAIR Principles

### Findability

- Clearly named experimental directories.
- Dedicated reproduction documentation.
- Timestamped experimental outputs.

### Accessibility

- Experimental artifacts stored alongside the code.
- Documented execution instructions.
- Preservation of the original repository structure.

### Interoperability

- Machine-readable CSV and JSON output formats.
- Standard Python analysis scripts.
- Explicit dependency versions.

### Reusability

- Reusable experimental runner.
- Recorded model and environment specifications.
- Documented code modifications.
- Independent accuracy verification.
- Reproducible statistical calculations.
- Separation of original and reproduced outputs.

These contributions support FAIR principles but
do not constitute a complete FAIR compliance assessment.

## 15. Conclusion

The Llama 3.1 8B zero-shot experiments on the
TREC Health Misinformation 2022 dataset were
successfully rerun.

The reproduction achieved accuracies of 88%,
82% and 84% for the no-context, non-expert and
expert configurations, respectively.

Although all three accuracies exceeded the
original reported values, none of the differences
were statistically significant.

The reproduction achieved 94.67% prediction
agreement with the original experiments and
preserved the original finding that the
no-context configuration achieved the highest
accuracy among these three configurations.

The findings demonstrate substantial agreement
with the original experiment, while exact
numerical reproducibility was not achieved.

Further repeated experiments would be necessary
to assess run-to-run stability and investigate
the causes of the observed differences.
