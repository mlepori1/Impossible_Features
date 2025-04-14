"""This file uses GPT-4 to generate novel sentence prefixes for the
steering vector evaluation
"""

from openai import OpenAI
import pandas as pd
import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outfile",
        default="../data/study_1_generated_prefixes.txt",
        type=str,
        help="File to put generated prefixes",
    )

    args = parser.parse_args()
    return args


def generate(n, prefixes):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Help me create a dataset of sentence prefixes. Here are some examples:\n{prefixes}\nPlease generate {str(n)} more examples for me. Please make sure every example ends with the words \"using a\"",
            },
        ],
    )
    return completion.choices[0].message.content.lower()


if __name__ == "__main__":

    args = parse_arguments()

    N = 60
    data = pd.read_csv("../data/shades_of_zero.csv")

    prefixes = ""
    for _, row in data.iterrows():
        prefix = row["classification_prefix"].replace("[POSS]", "their").strip() + "\n"
        prefixes += prefix
    prefixes = prefixes.strip()

    new_prefixes = generate(N, prefixes)
    outfile = open(args.outfile, "w")
    outfile.write(new_prefixes)
    outfile.close()
