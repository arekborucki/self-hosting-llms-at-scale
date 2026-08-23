from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

prompt = (
    "Invent one creative name for an internal AI assistant "
    "at a law firm. Answer with the name only."
)

for temperature in [0.1, 1.2]:
    print(f"\nTemperature {temperature}")
    for i in range(5):
        response = client.chat.completions.create(
            model="Qwen/Qwen3-4B",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        print(response.choices[0].message.content)
