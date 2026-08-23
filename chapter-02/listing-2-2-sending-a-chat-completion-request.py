from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1",
                api_key="not-needed-yet")

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[
        {"role": "system",
         "content": "You are a helpful assistantfor a law firm."},
        {"role": "user",
         "content": "In two sentences: what is an indemnification clause?"},
    ],
    max_tokens=128,
    extra_body={"chat_template_kwargs":{"enable_thinking": False}},
)
print(response.choices[0].message.content)
