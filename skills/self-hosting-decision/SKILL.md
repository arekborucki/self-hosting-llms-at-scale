---
name: self-hosting-decision
description: Helps decide whether an organization should self-host an open-weight LLM or keep using a hosted API. Use this skill whenever someone asks whether to self-host, run models on their own infrastructure, move off OpenAI or another API, or compare self-hosting with an API, even if they only mention privacy, cost, latency, compliance, or vendor lock-in.
---

# Deciding whether to self-host an LLM

Based on chapter 1 of *Self-Hosting LLMs at Scale* (Manning).

## Step 1: Check the five forces

Ask about each one. If any of them truly binds the organization, it usually settles the question in favor of self-hosting, whatever the practical signals in step 2 say.

1. **Privacy**: data that may not leave the organization (client documents, health records, source code).
2. **Cost at scale**: steady, high token volume where per-token API pricing becomes larger than the cost of GPUs that stay busy.
3. **Latency**: hard response-time requirements that a remote API cannot meet reliably.
4. **Vendor risk**: dependence on one provider's pricing, model changes, or availability.
5. **Regulation**: obligations that exist even with perfect privacy, such as auditability, where processing happens, sector rules, and the EU AI Act. Regulation covers what the organization must be able to prove, in addition to where the data goes.

## Step 2: Check the signals for staying on an API

Only if none of the five forces binds, a hosted API is probably the better choice when:

- volume is low or irregular, so a GPU would sit idle most of the time
- the team is still prototyping and the product is not proven yet
- the workload needs the strongest commercial models, and an open model is not good enough
- nobody will own and operate the platform, including incidents and upgrades
- no hard requirement forces control

## Step 3: Give a recommendation

Write a short answer with:
- the recommendation (self-host, stay on the API, or start on the API and plan a move)
- which force or signal decided it
- what would change the decision (for example, volume doubling, a new regulation)
- a next step: if self-hosting, estimate hardware with the gpu-memory-sizing skill; if staying on the API, design the integration against an OpenAI-compatible interface so a later move is cheap

Keep the tone neutral, and recommend the API when the signals point that way.
