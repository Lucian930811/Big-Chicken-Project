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
  - None (we used publicly available libraries/APIs, but wrote the project code ourselves).

Scripts/functions written by our team:
- setup_colab.ipynb (~56 lines)
        Sets up the Colab environment (auth tokens), logs into Hugging Face, and loads the Gemma model/tokenizer.
- sentiment_features.ipynb (~260 lines)
        Sentiment feature discovery pipeline: queries Neuronpedia/Gemma-Scope features using seed prompts,
        evaluates candidate SAE features on a sentiment dataset using AUC (roc_auc_score),
        and ranks/selects top sentiment-related features.
- sentiment_evaluation.ipynb (~580 lines)
        Evaluates sentiment effects (e.g., relative sentiment/reconstruction analyses) using model outputs,
        OpenAI API calls for scoring/labeling (if applicable), and visualization of results.
- IMDB_prefix.ipynb (~243 lines)
        Builds/constructs IMDB dataset prefixes (e.g., prompt/prefix generation and formatting),
        including dataset download/processing and optional transformer pipeline usage.
 - truthfulness_features.ipynb (~785 lines)
        Truthfulness feature discovery pipeline: constructs prompt sets (myth/fact/unanswerable),
        queries Neuronpedia for activated SAE features, aggregates statistics, and ranks candidates
        for truthfulness-related behavior.
- truthfulness_evaluation.ipynb (~408 lines)
        Runs truthfulness evaluation using selected features/prompts, performs scoring and plotting,
        and summarizes evaluation metrics and outcomes.
- steer.ipynb (~134 lines)
        Implements feature steering workflow using SAE Lens + TransformerLens (hooks/SAE loading),
        Plus inference utilities for testing steering effects on model generations.

