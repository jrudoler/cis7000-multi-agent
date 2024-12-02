from openai import AzureOpenAI
import os
from data_handling import load_data, sample_reviews

# Set up the OpenAI API client


def get_client():
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-01-preview",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return client


def example_prompts(reviews):
    """
    Generate a list of few-shot prompts based on the reviews.
    Assumes that the reviews have already been cleaned and have an 'item_description' column.
    """
    prompts = []
    for review in reviews:
        prompts.extend(
            [
                {"role": "user", "content": f"Item Description: {review['item_description']}"},
                {"role": "assistant", "content": f"Title: {review['title']}\nReview: {review['text']}"},
            ]
        )
    return prompts


def main(client, n_few_shot: int = 3, seed: int | None = None):
    reviews = load_data()
    # sample 3 reviews

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant that can review books based on the item metadata.",
    }
    examples = reviews.shuffle(seed=seed).take(n_few_shot + 1)
    user_prompt = {"role": "user", "content": examples.select([n_few_shot])["item_description"][0]}
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            system_prompt,
            *example_prompts(examples.select(range(n_few_shot))),
            user_prompt,
        ],
    )
    print(f"Given item description:\n{'-' * 100}\n{user_prompt['content']}\n{'-' * 100}\n")
    print(f"Response:\n{'-' * 100}\n{response.choices[0].message.content}\n{'-' * 100}\n")


if __name__ == "__main__":
    client = get_client()

    main(client)
