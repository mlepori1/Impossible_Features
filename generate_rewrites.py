from openai import OpenAI
from data import SentenceFeaturesDataset

def rewrite(client, sentence, topic):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Rewrite the following sentence to be about {topic}.\nSentence: {sentence}\n"
            }
        ]
    )

    print(f"{sentence} -> {completion.choices[0].message.content.lower()}")

    return completion.choices[0].message.content.lower()

def generate_rewrites(topic):
    client = OpenAI()
    dataset = SentenceFeaturesDataset(file="stimuli")
    rewrites = []

    for datum in dataset:
        rewrites.append(rewrite(client, datum["probable"], topic))

    dataset.data[topic] = rewrites
    return dataset.data

if __name__=="__main__":
    df = generate_rewrites("monsters")
    df.to_csv("./data/stimuli.csv")
