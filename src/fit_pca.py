"""
This file fits and saves an SKlearn PCA module to wikitext data,
in order to serve as a baseline for all evaluations 
"""

import os
from collections import defaultdict
import pickle as pkl
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import utils

from sklearn.decomposition import PCA

def process_dataset(model, tokenizer, dataset):
    """Extracts the final hidden state from every layer 
    for every sentence in dataset
    """
    states = defaultdict(list)
    for row in tqdm(
        dataset, total=len(dataset), desc="Processing wikitext"
    ):
        sentence = row["text"]
        if tokenizer.bos_token is not None:
            sentence = tokenizer.bos_token + sentence
        
        # Process sentence and extract hidden states of the last token
        inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
        all_states = model(**inputs, output_hidden_states=True).hidden_states
        for layer, layer_states in enumerate(all_states):
            last_tok_state = layer_states.cpu().float().numpy()[0, -1]
            states[layer].append(last_tok_state)

    return states

if __name__ == "__main__":
    config = utils.get_config()

    ### Set Up Output
    os.makedirs(config["artifact_path"], exist_ok=True)

    ### Set up model
    torch.set_grad_enabled(False)
    torch.manual_seed(19)
    np.random.seed(19)

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=utils.strtype2torchtype(config["dtype"]),
        device_map="auto",
        token="TOKEN",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        token="TOKEN",
    )
    ### Load wikitext dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1")["validation"]
 
    ### Extract hidden states
    states = process_dataset(model, tokenizer, dataset)

    ### Fit PCAs and save
    for layer in states.keys():
        pca = PCA(n_components=3)
        pca = pca.fit(np.stack(states[layer], axis=0))
        pkl.dump(pca, open(os.path.join(config["artifact_path"], f"Layer_{layer}_PCA.pkl"), "wb"))
