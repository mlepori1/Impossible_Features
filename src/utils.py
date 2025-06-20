import yaml
import argparse
import sys
from collections import defaultdict
import torch
from tqdm import tqdm
import numpy as np


def get_config():
    # Load config file from command line arg
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="where to load YAML configuration",
        metavar="FILE",
    )

    argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)

    if not hasattr(args, "config"):
        raise ValueError("Must include path to config file")
    else:
        with open(args.config, "r") as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)


def strtype2torchtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32


def compute_hidden_states(model, tokenizer, split, feature_label):
    """Extract the hidden states of the final token (".")
    from every sentence exhibiting a particular feature from
    every split.
    """
    layer2states = defaultdict(dict)
    split = split[split["label"] == feature_label]

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Computing Hidden States"):
        sentence = row["sentence"]
        item_set_id = row["item_set_id"]
        if tokenizer.bos_token is not None:
            sentence = tokenizer.bos_token + sentence

        # Process sentence and extract hidden states of the last token
        inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
        all_states = model(**inputs, output_hidden_states=True).hidden_states
        for layer, layer_states in enumerate(all_states):
            # Asserts to ensure that input is processed correctly
            assert len(layer_states.shape) == 3
            assert layer_states.shape[0] == 1
            assert layer_states.shape[1] == len(inputs.input_ids[0])

            last_tok_state = layer_states.cpu().float().numpy()[0, -1].reshape(-1)
            layer2states[layer][item_set_id] = last_tok_state

    return layer2states


def compute_summed_log_probs_for_classification(model, tokenizer, split, feature_label):
    """Compute the summed log prob of every sentence exhibiting a
    particular modal feature.
    """
    all_probabilities = {}
    split = split[split["label"] == feature_label]

    for _, row in tqdm(split.iterrows(), total=len(split), desc="Computing Log Probs"):
        sentence = row["sentence"]
        item_set_id = row["item_set_id"]
        if tokenizer.bos_token is not None:
            sentence = tokenizer.bos_token + sentence

        ipts = tokenizer(sentence, return_tensors="pt").to("cuda")
        logits = model(**ipts).logits
        # Asserts to ensure that input is processed correctly
        assert len(logits.shape) == 3
        assert logits.shape[0] == 1
        assert logits.shape[1] == len(ipts.input_ids[0])

        # Get the log probability of each token from the 2nd to the last, according
        # to the previous residual stream.
        log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)[:-1]
        tokens = ipts.input_ids[0, 1:]

        summed_prob = (
            torch.sum(log_probs[torch.arange(len(tokens)), tokens]).float().cpu().numpy()
        )
        all_probabilities[item_set_id] = summed_prob

    return all_probabilities


def classify_vector(higher_state, lower_state, vector, unit_norm=False):
    """Helper function to classify two hidden states based on projections
    along a vector
    """

    if np.linalg.norm(vector) == 0:
        return 0, 0.0

    if unit_norm:
        # Norm every vector in the projection
        vector = vector / np.linalg.norm(vector)
        higher_state = higher_state / np.linalg.norm(higher_state)
        lower_state = lower_state / np.linalg.norm(lower_state)

    high_proj = np.dot(higher_state, vector)
    low_proj = np.dot(lower_state, vector)

    correct_bool = high_proj > low_proj
    delta = high_proj - low_proj

    return correct_bool, delta


def project_data(data, model, tokenizer, vector_tuples):
    """Extract the relevant hidden states of the final token (".")
    from every sentence, project onto each vector.

    vector_tuples is comprised of (vector, layer) combinations
    """
    projections = [[] for _ in range(len(vector_tuples))]

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Projecting data"):
        sentence = row["sentence"]
        if tokenizer.bos_token is not None:
            sentence = tokenizer.bos_token + sentence

        # Process sentence and extract hidden states of the last token
        inputs = tokenizer(sentence, return_tensors="pt").to("cuda")
        all_states = model(**inputs, output_hidden_states=True).hidden_states

        # Project the final hidden state representation (at the appropriate layer
        # onto each of the provided vectors
        for vect_idx in range(len(vector_tuples)):
            vector, layer = vector_tuples[vect_idx]
            final_state = all_states[layer].cpu().float().numpy()[0, -1].reshape(-1)
            proj = np.dot(final_state, vector).item()
            projections[vect_idx].append(proj)

    return projections


def compute_summed_log_probs(data, model, tokenizer):
    """Compute the summed log prob of every sentence
    """
    probabilities = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Computing Log Probs"):
        sentence = row["sentence"]
        if tokenizer.bos_token is not None:
            sentence = tokenizer.bos_token + sentence

        ipts = tokenizer(sentence, return_tensors="pt").to("cuda")
        probs = torch.nn.functional.log_softmax(model(**ipts).logits[0], dim=-1)[:-1]
        tokens = ipts.input_ids[0, 1:]
        summed_prob = (
            torch.sum(probs[torch.arange(len(tokens)), tokens]).float().cpu().numpy()
        )
        probabilities.append(summed_prob)

    return probabilities
