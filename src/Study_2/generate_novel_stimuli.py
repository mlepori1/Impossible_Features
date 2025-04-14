from openai import OpenAI
import pandas as pd

def generate(n, feature, description, sentences):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Help me create a dataset of {feature} scenarios: {description}. Here are some examples:\n{sentences}\nPlease generate {str(n)} more examples for me. Examples should end with a period and should be separated with a newline."
            }
        ]
    )

    print(completion.choices[0].message.content.lower())

    return completion.choices[0].message.content.lower()

def get_examples(data, feature, n):
    # Get N random examples
    examples = ""
    rows = data.sample(n)
    for idx in range(n):
        row = rows.iloc[idx]
        prefix = row["classification_prefix"].replace("[POSS]", "their").strip()
        cont = row[feature].replace("[POSS]", "their").strip()
        example = prefix + " " + cont + ".\n"
        examples += example
    return examples.strip()


if __name__=="__main__":

    TOTAL_SAMPLES = 200
    PER_BATCH = 40
    TOTAL_BATCHES = int(TOTAL_SAMPLES/PER_BATCH)

    data = pd.read_csv("./data/stimuli_with_syntax.csv")

    improb_file = open("./data/experiment_3/generated_improbable.txt", "w")
    imposs_file = open("./data/experiment_3/generated_impossible.txt", "w")
    inc_file = open("./data/experiment_3/generated_inconceivable.txt", "w")

    improb_desc = "scenarios that are possible in the real world, but are merely unlikely"
    imposs_desc = "scenarios that one can imagine happening in some universe, but cannot happen in our world because they violate the laws of physics"
    inc_desc = "scenarios that do not make sense due to some basic conceptual error"

    for _ in range(TOTAL_BATCHES):
        improbable_examples = get_examples(data, "improbable", 5)
        impossible_examples = get_examples(data, "impossible", 5)
        nonsensical_examples = get_examples(data, "inconceivable", 5)

        improbable_generations = generate(PER_BATCH, "improbable", improb_desc, improbable_examples)
        improb_file.write(improbable_generations + "\n")

        impossible_generations = generate(PER_BATCH, "impossible", imposs_desc, impossible_examples)
        imposs_file.write(impossible_generations + "\n")

        nonsensical_generations = generate(PER_BATCH, "nonsensical", inc_desc, nonsensical_examples)
        inc_file.write(nonsensical_generations + "\n")

    improb_file.close()
    imposs_file.close()
    inc_file.close()

