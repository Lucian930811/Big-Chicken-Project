# Big-Chicken-Project
CS 175 Project - Mechanistic Interpretability

Libraries used: 
- numpy
    - pandas
    - matplotlib
    - seaborn
    - tqdm / tqdm.asyncio
    - requests
    - scikit-learn (sklearn.metrics)
    - datasets (Hugging Face Datasets)
    - transformers (Hugging Face Transformers)
    - huggingface_hub
    - torch / torchvision / torchaudio
    - transformer_lens
    - sae_lens
    - openai (OpenAI API client)
    - google.colab (drive, userdata)

Publicly available codes used:
N/A(we used publicly available libraries/APIs, but wrote the project code ourselves).

Scripts/functions written by our team:
- setup_colab.ipynb (~56 lines)
        Tests colab environment, hugging face, and loads gemma model. Week 1 testing.
- sentiment_features.ipynb (~260 lines)
        Script for finding sentiment-related feature. It sends polarized seed prompts to Neruonpedia to see what activates, evaluates the candidate SAE features using AUC, and ranks the best ones for our sentiment steering.        and ranks/selects top sentiment-related features.
- IMDB_prefix.ipynb (~243 lines)
        Script to prepare the IMDB dataset. We used it to download the preprocess the movie reviews to generate the neutral prefixes that we needed for the sentiment steering experiments.
 - truthfulness_features.ipynb (~785 lines)
        Pipeline for finding truthfulness features. Constructs set of myth, fact, and unanswerable prompts, pulls activated features from Neuronpedia, and calculates stats to find features that control factual behavior.
- steer.ipynb (~134 lines)
        Core logic for model steering and mechanistic intervention. Uses SAE Lens and TransformerLens to load the autoencoders and inject our chosen feature vectors into the model. Used primarily prior to intermediate report for intial findings and testing.
- cs175_final_runs.ipynb (~480 lines)
        Notebook for all final full generations and experiment runs. Uses the same logic as in steer.ipynb, saves output for use in evaluation. Also includes the script we used for evaluating feature entanglement and calculating experiment sample size.
- sentiment_evaluation.ipynb (~580 lines)
        Evaluates output for sentiment steering sweep and subsequent experiments. Uses LLM-as-a-Judge to score generated texts, analyzes relative sentiment and coherence, and creates visualizations for report.
        OpenAI API calls for scoring/labeling (if applicable), and visualization of results.
- truthfulness_evaluation.ipynb (~408 lines)
        Evaluates output from the truthfulness steering sweep. Uses a different pipline to score the model's responses, calculate final metrics, and generate plots for our report.

