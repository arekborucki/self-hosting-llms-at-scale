stream = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user",
               "content": "List three risks in cross-border M&A."}],
    max_tokens=256,
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "",end="", flush=True)
