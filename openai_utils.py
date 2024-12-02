from openai import OpenAI


def get_key():
    return "<open AI key>" 


def get_GPT_response(prompt_sys, prompt_user, model = "gpt-4o", temp = 0):


    client = OpenAI(api_key= get_key())

    response = client.chat.completions.create(
        messages=[
            {
            "role": "system",
            "content": prompt_sys
            },
            {
            "role": "user",
            "content": prompt_user
            }
        ],
        model=model,
        temperature=temp
    )
    api_response = response.choices[0].message.content



    return api_response

