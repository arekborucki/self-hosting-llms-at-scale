---
name: model-selection
description: Runs a repeatable process for choosing an open-weight LLM for a specific use case: license check, sizing check, a golden set of test questions judged against written criteria, and a decision file. Use this skill whenever someone asks which open model to use, how to compare models, how to evaluate a model for their own work, whether a model's license allows commercial use, or how to document a model choice, even if they only ask "which model is best".
---

# Choosing an open-weight model

Based on chapter 4 of *Self-Hosting LLMs at Scale* (Manning).

## The process

1. **Discover candidates**: leaderboards, model announcements, and model families with a good track record. This step only produces names.
2. **Check the license**: read the actual license text. Note commercial use, user limits, acceptable-use restrictions, and attribution rules. A label such as "open" is not enough. Disqualify early.
3. **Check the size**: apply the memory equation (gpu-memory-sizing skill) against the hardware budget, which is usually a short list: the card already in use and one upgrade option.
4. **Run acceptance tests on a golden set** (below).
5. **Record the decision** (below).

## The golden set

- 20 or so questions taken from the real workload, written with the people who will use the model.
- For each question, write the pass criteria **before** seeing any answers.
- Include a few traps: questions where the correct answer is to refuse, to say the information is missing, or to avoid inventing a source.
- Run every candidate under identical conditions: same system message, temperature 0, same output limit, same template settings.
- Judge blind: hide model names while reviewing. Record pass or fail per question and a short note.
- Look at metadata too: unusually long or unusually short answers often point to the interesting cases.

## The decision file

One short file next to the deployment configuration with:
- the chosen model and its exact revision
- the license and the date it was checked
- golden set results for every tested candidate
- the tested alternative and why it was not chosen
- open questions and the date of the next re-evaluation

## Output

A short comparison table (candidate, license, fits, golden set score, notes), a recommendation, and a draft decision file. Do not recommend a model without its golden set results; if the user has not run the tests, help them build the golden set first.
