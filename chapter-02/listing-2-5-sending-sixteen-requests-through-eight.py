import concurrent.futures
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

def ask(i):
    r = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user",
                   "content": f"Question {i}: define force majeure "
                              "in one sentence."}],
        max_tokens=64,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.choices[0].message.content

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    for result in ex.map(ask, range(16)):
        print(result[:60])
