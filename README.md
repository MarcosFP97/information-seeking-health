# Evaluation of Generative Models Reliability to Obtain Medical Advice

This repository contains the code and results from the experiments reported in the paper "*Are Large Language Models Reliable to Obtain Medical Advice?
A Zero-Shot and a Few-Shot Evaluation*''.

## Important notes

- Prompts have been named slighlty differently when conducting the experimentation and when reporting them in the paper. Thus, the experiments named as "*expert*" correspond to the runs called "*h2oloo*", while those named as "*non-expert*" in the manuscript correspond to the runs, "*usc*".

- It is also important to notice that, since the accuracy was **manually revised**, it could happen that the printed accuracy in the log does not correspond with the real accuracy. To obtain the real accuracy, reported in the paper, you need to count the number of hits in each log file (the test McNemar file can help with that).
