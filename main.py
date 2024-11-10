from openai import AzureOpenAI
import os
from data_handling import load_data, clean_review, clean_meta

# Set up the OpenAI API client


def get_client():
    client = AzureOpenAI(
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-08-01-preview",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )
    return client


def main():
    dataset = load_data()
    dataset = load_data(review_filter=clean_review, meta_filter=clean_meta)
    # sample 3 reviews
    selected_reviews = reviews.shuffle(seed=42).take(3)
    selected_meta = [
        concat_item_metadata(all_meta[_])["cleaned_metadata"]
        for _ in selected_reviews.select_columns(["parent_asin"]).to_dict()["parent_asin"]
    ]
    selected_reviews = selected_reviews.add_column("item_description", selected_meta)


if __name__ == "__main__":
    client = get_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, how can I use Azure OpenAI?"}],
    )
    print(print(response.choices[0].message.content))
