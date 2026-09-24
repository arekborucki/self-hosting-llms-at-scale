"""Review golden-set answers side by side, blind, in the terminal.

Reads the answer files written by the golden-set harness (listing 4.1),
one file per candidate model, and asks you to mark every answer as
pass or fail against the criteria. Model names are hidden while you
judge: candidates appear as A, B, C in a new random order for every
question. Names are revealed in the results.

Usage:
    python review_answers.py answers_Qwen3-4B.jsonl answers_Qwen3-8B.jsonl

Keys at each answer:
    p = pass, f = fail, s = skip, n = add a note, q = save and quit

Verdicts are saved to golden_set_verdicts.csv after every answer, so you
can quit at any time and run the same command again to continue.
"""
import csv
import json
import os
import random
import sys
import textwrap

OUT = "golden_set_verdicts.csv"
WIDTH = 78


def load_answers(path):
    name = os.path.basename(path)
    name = name.removeprefix("answers_").removesuffix(".jsonl")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return name, {str(r["id"]): r for r in rows}, [str(r["id"]) for r in rows]


def load_verdicts():
    if not os.path.exists(OUT):
        return {}
    with open(OUT, newline="") as f:
        return {(r["id"], r["model"]): r for r in csv.DictReader(f)}


def save_verdicts(verdicts):
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "model", "verdict", "note"])
        w.writeheader()
        for (qid, model), r in sorted(verdicts.items()):
            w.writerow({"id": qid, "model": model,
                        "verdict": r["verdict"], "note": r["note"]})


def show(label, text):
    print(f"\n{label}")
    for para in str(text).splitlines() or [""]:
        print(textwrap.fill(para, WIDTH, initial_indent="  ",
                            subsequent_indent="  ") or "  ")


def ask(prompt, allowed):
    while True:
        key = input(prompt).strip().lower()
        if key in allowed:
            return key
        print(f"  Type one of: {', '.join(allowed)}")


def review(models, ids, verdicts):
    for n, qid in enumerate(ids, 1):
        pending = [m for m in models
                   if (qid, m[0]) not in verdicts]
        if not pending:
            continue
        ref = next(m[1][qid] for m in models if qid in m[1])
        order = random.sample(models, len(models))
        print("\n" + "=" * WIDTH)
        print(f"Question {n} of {len(ids)}  ({qid})")
        show("Prompt:", ref["prompt"])
        show("Pass if:", ref["criteria"])
        for pos, (name, rows, _) in enumerate(order):
            if (qid, name) in verdicts:
                continue
            row = rows.get(qid)
            label = f"Answer {'ABCDEFGH'[pos]}"
            if row is None:
                show(label + ":", "(no answer in this file)")
            else:
                meta = f"{row.get('tokens', '?')} tokens, " \
                       f"{row.get('seconds', '?')} s"
                show(f"{label} ({meta}):", row["answer"])
            key = ask("  [p]ass [f]ail [s]kip [n]ote [q]uit: ",
                      ["p", "f", "s", "n", "q"])
            if key == "q":
                return False
            if key == "s":
                continue
            note = ""
            if key == "n":
                note = input("  Note: ").strip()
                key = ask("  [p]ass or [f]ail: ", ["p", "f"])
            verdicts[(qid, name)] = {
                "verdict": "pass" if key == "p" else "fail",
                "note": note,
            }
            save_verdicts(verdicts)
    return True


def results(models, ids, verdicts):
    print("\n" + "=" * WIDTH)
    print("Results (model names revealed)\n")
    print(f"  {'Candidate':<24}{'Passed':>10}{'Failed':>10}{'Open':>8}")
    for name, _, _ in models:
        got = [verdicts.get((qid, name), {}).get("verdict") for qid in ids]
        passed, failed = got.count("pass"), got.count("fail")
        print(f"  {name:<24}{passed:>6} / {len(ids):<3}"
              f"{failed:>8}{len(ids) - passed - failed:>8}")
    print(f"\nVerdicts saved in {OUT}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    models = [load_answers(p) for p in sys.argv[1:]]
    ids = []
    for _, _, order in models:
        ids += [qid for qid in order if qid not in ids]
    verdicts = load_verdicts()
    finished = review(models, ids, verdicts)
    if finished:
        results(models, ids, verdicts)
    else:
        print(f"\nSaved. Run the same command again to continue.")


if __name__ == "__main__":
    main()
