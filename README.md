# Cross-Domain Recommendation Meets Large Language Models

### Abstract

Cross-domain recommendation (CDR) has emerged as a pro\-mising solution to the cold-start problem, faced by single-domain recommender systems. However, existing CDR models rely on complex neural architectures, large datasets, and significant computational resources, making them less effective in data-scarce scenarios or when simplicity is crucial. In this work, we leverage the reasoning capabilities of large language models (LLMs) and explore their performance in the CDR domain across multiple domain pairs. We introduce two novel prompt designs tailored for CDR and demonstrate that LLMs, when prompted effectively, outperform state-of-the-art CDR baselines across various metrics and domain combinations in the rating prediction and ranking tasks. This work bridges the gap between LLMs and recommendation systems, showcasing their potential as effective cross-domain recommenders.

## Project Files Information

You will see three main folders and files in this project repository:

1. **`dataset`**: This folder will contain all the datasets needed for the project. 
   - Before running this code, you need to place the necessary datasets into this folder in order to get results.
     - Go to this website and download the metadata and reviews dataset for each domain: [Amazon Reviews Dataset](https://jmcauley.ucsd.edu/data/amazon/).
     - Choose two domains for Cross-Domain Recommendation (CDR) and place the necessary datasets in this folder. 
       - If you want to use **Books** as the source and **Movies** as the target, place the following files in the **`dataset`** folder:
         1. `meta_Books.json`
         2. `reviews_Books_5.json`
         3. `meta_Movies_and_TV.json`
         4. `reviews_Movies_and_TV_5.json`
    
  2. **`src`**: Contains all the source code files required to run experiments in the project. The files in this folder are:
     - `process_data.py`: Takes the source and target domain data from the `datasets` folder and pre-processes it for CDR.
     - `ranking_prediction_prompts.py`: Contains all the prompts required for the ranking task in CDR that we use for the LLM.
     - `rating_prediction_prompts.py`: Contains all the prompts required for the rating prediction task in CDR that we use for the LLM. 
     - `evaluate.py`: Evaluates LLama models (llama-2-7b-chat, llama-2-13b-chat, and llama-3-8b-instruct) in rating and ranking tasks.
     - `gpt_evaluate.py`: Evalures GPT models (GPT-3.5-turbo, GPT-4, and GPT-4o) in the rating and ranking tasks for CDR. 
     - `run.py`: The most important file that connects all the source files together. You will only need to run this file to get results.

 3. **`requirements.txt`**: Contains all the libraries and packages used to run this project. :
     - Create a python virual enviroment with this command: `python3 -m venv llm_environment`.
     - Once you create the virual enviroment, activate it with this command: `source llm_environment/bin/activate`.
     - Finally, run this command to download all the neccessary libraries required: `pip install -r requirements.txt`

## Running the Code


