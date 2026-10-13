"""Build EN-JA parallel pairs from Tatoeba (CC-BY 2.0 FR): links join.
Writes data/pairs.jsonl {ja, en} (cap 20k, len 10-200 chars each)."""
import csv

JPN, ENG, CAP = set(), {}, 20000


def main() -> None:
    with open("data/jpn_sentences.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and 10 <= len(p[2]) <= 200:
                JPN.add(p[0])
    with open("data/eng_sentences.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and 10 <= len(p[2]) <= 200:
                ENG[p[0]] = p[2]
    pairs, seen = [], set()
    with open("data/links.csv", encoding="utf-8") as f:
        for a, b in csv.reader(f, delimiter="\t"):
            if a in JPN and b in ENG and (a, b) not in seen:
                seen.add((a, b))
                pairs.append((a, b))
            elif b in JPN and a in ENG and (b, a) not in seen:
                seen.add((b, a))
                pairs.append((b, a))
            if len(pairs) >= CAP:
                break
    import json

    with open("data/jpn_sentences.tsv", encoding="utf-8") as f:
        texts = {p[0]: p[2] for line in f for p in [line.rstrip("\n").split("\t")] if len(p) == 3}
    with open("data/pairs.jsonl", "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(json.dumps({"ja": texts[a], "en": ENG[b]}, ensure_ascii=False) + "\n")
    print("pairs:", len(pairs))


if __name__ == "__main__":
    main()
