"""Лабораторная 2 · Трансформер руками: токены, векторы, внимание.

Эталонное решение. Запуск:

    python run.py              # полный прогон
    python run.py --fast       # режим для CI: меньше слияний, короче контексты
    python run.py --skip-model # только часть 1, без загрузки модели

Пути считаются от папки скрипта, запускать можно откуда угодно. Корпус скачивается
в corpus.txt рядом со скриптом, артефакты пишутся в artifacts/: bpe.json,
tokens_per_word.json, paragraphs.json, search.json, attention_map.json,
attention_map.png, dilution.csv. Модели и корпус можно переопределить флагами
--model, --e5, --corpus или переменными LAB_MODEL, LAB_E5_MODEL, LAB_CORPUS.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"
WORD = re.compile(r"[а-я]+(?:-[а-я]+)*")

# ----------------------------------------------------------------------------
# Общее
# ----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_raw(path: Path) -> str:
    """Корпус как есть, с регистром: нужен для абзацев, которые читает модель."""
    if not path.exists():
        log(f"скачиваю корпус в {path}")
        urllib.request.urlretrieve(CORPUS_URL, path)
    return path.read_text(encoding="utf-8")


def normalize(raw: str) -> str:
    return raw.lower().replace("ё", "е")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Часть 1. Свой BPE
# ----------------------------------------------------------------------------

END = "</w>"


def train_bpe(word_freq: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Классический BPE по словарю слов с частотами. Возвращает список слияний по порядку."""
    words = {tuple(w) + (END,): f for w, f in word_freq.items()}
    merges: list[tuple[str, str]] = []
    for step in range(num_merges):
        pairs: Counter = Counter()
        for sym, f in words.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        a, b, ab = best[0], best[1], best[0] + best[1]
        new_words = {}
        for sym, f in words.items():
            if a in sym and b in sym:
                out, i = [], 0
                while i < len(sym):
                    if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                        out.append(ab); i += 2
                    else:
                        out.append(sym[i]); i += 1
                sym = tuple(out)
            new_words[sym] = f
        words = new_words
        if (step + 1) % 100 == 0:
            log(f"  слияний: {step + 1}, последнее {a!r}+{b!r} ({pairs[best]})")
    return merges


def apply_bpe(word: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    sym = list(word) + [END]
    while len(sym) > 1:
        candidates = [(ranks[(a, b)], i) for i, (a, b) in enumerate(zip(sym, sym[1:])) if (a, b) in ranks]
        if not candidates:
            break
        _, i = min(candidates)
        sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
    return sym


def part_bpe(text: str, out: Path, fast: bool) -> None:
    log("часть 1: свой BPE")
    words = WORD.findall(text)
    freq = Counter(words)
    min_freq = 5 if fast else 2
    num_merges = 200 if fast else 1500
    train_freq = {w: f for w, f in freq.items() if f >= min_freq}
    log(f"  слов {len(words):,}, уникальных {len(freq):,}, в обучении {len(train_freq):,}")
    merges = train_bpe(train_freq, num_merges)
    ranks = {m: i for i, m in enumerate(merges)}
    vocab = set(ch for w in train_freq for ch in w) | {END} | {a + b for a, b in merges}

    cache: dict[str, int] = {}
    n_tokens = 0
    for w, f in freq.items():
        if w not in cache:
            cache[w] = len(apply_bpe(w, ranks))
        n_tokens += cache[w] * f
    n_chars = sum(len(w) * f for w, f in freq.items())
    stats = {
        "chars": n_chars,
        "words": len(words),
        "tokens_bpe": n_tokens,
        "tokens_per_word": round(n_tokens / len(words), 3),
        "compression_vs_chars": round(n_chars / n_tokens, 3),
    }
    log(f"  словарь {len(vocab):,}, токенов {n_tokens:,}, сжатие против символов {stats['compression_vs_chars']}")
    examples = {w: apply_bpe(w, ranks) for w in ["наполеон", "андрей", "сказала", "неудовольствие", "лошадь"] if w in freq}
    json.dump({"vocab_size": len(vocab), "num_merges": len(merges), "merges": merges, "stats": stats, "examples": examples},
              open(out / "bpe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------
# Часть 2. Токенизатор настоящей модели
# ----------------------------------------------------------------------------

EN_PARAGRAPH = (
    "The prince had been talking for almost an hour, and the guests were growing tired. "
    "Anna looked at the clock, then at the window, then at the clock again. Nobody dared to leave "
    "before the host, and the host did not seem to notice the time at all. A servant brought tea, "
    "and the conversation turned, as it always did in that house, to the war and to the emperor."
)


CONTEXT_TOKENS = 4096


def count_tokens(tok, text: str) -> int:
    """Сколько токенов займет текст у модели. Та же функция пригодится в сервисе для учета расхода."""
    return len(tok(text, add_special_tokens=False)["input_ids"])


def part_tokenizer(text: str, tok, model_id: str, out: Path) -> None:
    log("часть 2: токенизатор модели и бюджет контекста")
    ru_words = WORD.findall(text)
    ru_paragraph = " ".join(ru_words[500:600])
    res = {"tokenizer": model_id, "vocab_size": len(tok), "context_tokens": CONTEXT_TOKENS}
    for lang, par in (("ru", ru_paragraph), ("en", EN_PARAGRAPH)):
        n_words = len(par.split())
        n_tok = count_tokens(tok, par)
        tpw = n_tok / n_words
        res[lang] = {"words": n_words, "tokens": n_tok, "tokens_per_word": round(tpw, 3),
                     "words_in_context": int(CONTEXT_TOKENS / tpw)}
        log(f"  {lang}: {n_words} слов, {n_tok} токенов, {tpw:.2f} на слово, "
            f"в контекст {CONTEXT_TOKENS} влезает {res[lang]['words_in_context']} слов")
    res["example_ru"] = tokenize_with_spans(tok, " ".join(ru_words[500:512]))[2]
    json.dump(res, open(out / "tokens_per_word.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------
# Часть 3. Векторы и поиск по корпусу
# ----------------------------------------------------------------------------

E5_MODEL = "intfloat/multilingual-e5-small"
TOP_K = 5
QUERIES = [
    ("Хозяйка вечера несколько дней болела гриппом и утром разослала приглашения", "кашляла несколько дней"),
    ("Отец говорит, что оба его сына дураки, только один спокойный, а другой беспокойный", "покойный дурак"),
    ("Богатый и скупой старый князь живет в деревне, его прозвали прусским королем", "прозванный прусским королем"),
    ("Гости по очереди здоровались с никому не нужной тетушкой", "никому не интересной и не нужной тетушки"),
    ("Брат удивительно похож на сестру-красавицу, но при этом очень некрасив", "поразительно дурен собой"),
    ("В гостиную входит невысокий молодой князь с усталым и скучающим взглядом", "усталого, скучающего взгляда"),
    ("Обедневшая княгиня приехала просить, чтобы ее сына перевели в гвардию", "выхлопотать определение в гвардию"),
    ("Спор о революции: народ отдал Бонапарту власть, чтобы избавиться от Бурбонов", "избавил его от Бурбонов"),
    ("Неуклюжий толстый гость по рассеянности взял чужую генеральскую шляпу", "треугольную шляпу с генеральским плюмажем"),
    ("Мальчика десяти лет отправили за границу с гувернером-аббатом", "гувернером-аббатом за границу"),
    ("Совет другу: не женись, пока не сделал всего, что мог", "не женись до тех пор"),
    ("Светлая петербургская ночь, второй час, он едет домой и не может заснуть", "бессумрачная ночь"),
]


def select_paragraphs(raw: str, limit: int) -> list[str]:
    """Абзацы по переводам строки, длиной от 40 до 120 слов, первые limit штук."""
    out = []
    for p in raw.split("\n"):
        p = p.strip()
        n = len(WORD.findall(normalize(p)))
        if 40 <= n <= 120:
            out.append(p)
            if len(out) == limit:
                break
    return out


def embed_static(tok, model, texts: list[str]):
    """Среднее входных эмбеддингов по токенам: таблица, контекста нет."""
    import torch
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    vecs = [emb[tok(t, add_special_tokens=False)["input_ids"]].mean(0) for t in texts]
    return torch.stack(vecs)


def embed_contextual(tok, model, texts: list[str], device: str):
    """Среднее скрытых состояний последнего слоя: те же токены, но после внимания."""
    import torch
    vecs = []
    with torch.no_grad():
        for t in texts:
            enc = tok(t, return_tensors="pt", add_special_tokens=False).to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1][0]
            vecs.append(h.float().mean(0).cpu())
    return torch.stack(vecs)


def embed_e5(e5_tok, e5, texts: list[str], prefix: str, device: str):
    """Энкодер, обученный под поиск: среднее по маске, префиксы query:/passage:."""
    import torch
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = [prefix + t for t in texts[i:i + 16]]
            enc = e5_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            h = e5(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            vecs.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(vecs)


def search(q_vecs, p_vecs, k: int) -> list[list[int]]:
    """Для каждого запроса k ближайших абзацев по косинусу."""
    import torch
    q = q_vecs / q_vecs.norm(dim=1, keepdim=True)
    p = p_vecs / p_vecs.norm(dim=1, keepdim=True)
    return torch.topk(q @ p.T, k, dim=1).indices.tolist()


def part_search(raw: str, tok, model, device: str, out: Path, fast: bool, e5_id: str) -> None:
    from transformers import AutoModel, AutoTokenizer
    log("часть 3: векторы и поиск по корпусу")
    paragraphs = select_paragraphs(raw, 60 if fast else 500)
    log(f"  абзацев {len(paragraphs)}")
    json.dump([{"id": i, "text": p} for i, p in enumerate(paragraphs)],
              open(out / "paragraphs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    queries = [q for q, _ in QUERIES]
    answers = []
    for _, phrase in QUERIES:
        hit = [i for i, p in enumerate(paragraphs) if phrase in p]
        answers.append(hit[0] if hit else -1)

    e5_tok = AutoTokenizer.from_pretrained(e5_id)
    e5 = AutoModel.from_pretrained(e5_id).to(device).eval()
    tops = {
        "static": search(embed_static(tok, model, queries), embed_static(tok, model, paragraphs), TOP_K),
        "contextual": search(embed_contextual(tok, model, queries, device),
                             embed_contextual(tok, model, paragraphs, device), TOP_K),
        "e5": search(embed_e5(e5_tok, e5, queries, "query: ", device),
                     embed_e5(e5_tok, e5, paragraphs, "passage: ", device), TOP_K),
    }
    del e5
    recall = {m: round(sum(a in t for a, t in zip(answers, tops[m])) / len(QUERIES), 3) for m in tops}
    for m, r in recall.items():
        log(f"  recall@{TOP_K} {m}: {r}")
    json.dump({"k": TOP_K, "n_paragraphs": len(paragraphs),
               "queries": [{"query": q, "answer_phrase": ph, "answer_id": a,
                            "top": {m: tops[m][i] for m in tops}}
                           for i, ((q, ph), a) in enumerate(zip(QUERIES, answers))],
               "recall_at_k": recall},
              open(out / "search.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ----------------------------------------------------------------------------
# Часть 4. Внимание
# ----------------------------------------------------------------------------

PRONOUN_SENTENCE = "Врач осмотрел пациента и сказал, что его анализы готовы."
PRONOUN = "его"
ANTECEDENT = "пациента"
INSTRUCTION = "Отвечай только по-русски и в конце каждого ответа ставь слово КОНЕЦ."


def attentions(model, ids: list[int], device: str):
    """Прогоняет модель по списку id и возвращает кортеж карт внимания по слоям, каждая [heads, n, n] на CPU."""
    import torch
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=x, output_attentions=True)
    return tuple(a[0].float().cpu() for a in out.attentions)


def tokenize_with_spans(tok, text: str):
    """id токенов и подписи к ним: подпись это кусок исходного текста, который токен покрывает.
    Так подписи остаются читаемыми, даже если токен это половина буквы в байтах."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    labels = [text[s:e] for s, e in offsets]
    return ids, offsets, labels


def span(offsets: list[tuple[int, int]], text: str, word: str) -> list[int]:
    """Позиции токенов, которые покрывают первое вхождение слова."""
    start = text.find(word)
    if start < 0:
        return []
    end = start + len(word)
    return [i for i, (s, e) in enumerate(offsets) if e > start and s < end]


def part_attention_map(model, tok, device: str, out: Path) -> None:
    log("часть 4а: карта внимания для местоимения")
    ids, offsets, tokens = tokenize_with_spans(tok, PRONOUN_SENTENCE)
    import torch
    attn = torch.stack(attentions(model, ids, device))   # [layers, heads, n, n]
    p = span(offsets, PRONOUN_SENTENCE, PRONOUN)
    a = span(offsets, PRONOUN_SENTENCE, ANTECEDENT)
    if not p or not a:
        log("  местоимение или антецедент не найдены в токенах, пропускаю карту")
        return
    row = attn[:, :, p[-1], :]                       # [layers, heads, n]
    score = row[:, :, a].sum(-1)                      # внимание на антецедент
    layer, head = divmod(int(score.argmax()), score.shape[1])
    weights = row[layer, head].tolist()
    log(f"  лучшая голова: слой {layer}, голова {head}, доля на «{ANTECEDENT}» {score[layer, head]:.2f}")
    json.dump({"sentence": PRONOUN_SENTENCE, "tokens": tokens, "pronoun_pos": p, "antecedent_pos": a,
               "best_layer": layer, "best_head": head, "weights": weights,
               "share_on_antecedent": round(float(score[layer, head]), 4)},
              open(out / "attention_map.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.bar(range(len(tokens)), weights, color="#333333")
        ax.set_xticks(range(len(tokens))); ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
        ax.set_title(f"На что смотрит «{PRONOUN}»: слой {layer}, голова {head}", fontsize=10)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "attention_map.png", dpi=150); plt.close(fig)
    except Exception as e:  # картинка необязательна
        log(f"  картинка не сохранена: {e}")


def part_dilution(text: str, model, tok, device: str, out: Path, fast: bool) -> None:
    log("часть 4б: размывание внимания")
    # Карты внимания всех слоев при n=1536 у Qwen3-0.6B занимают около 4 ГБ, поэтому длины умеренные.
    lengths = [64, 256, 768] if fast else [128, 512, 1024]
    prefix_ids = tok(INSTRUCTION + "\n", add_special_tokens=False)["input_ids"]
    filler_ids = tok(" ".join(WORD.findall(text)[20000:30000]), add_special_tokens=False)["input_ids"]
    # Первый токен входа это сток (attention sink): модель сбрасывает на него около половины внимания при любом тексте.
    # Если считать его частью инструкции, кривая показывает сток, а не инструкцию, поэтому он идет отдельной колонкой.
    # Последний слой без стока шумит, среднее по всем слоям убывает монотонно.
    instr = slice(1, len(prefix_ids))
    instr_len = len(prefix_ids) - 1
    rows = []
    for n in lengths:
        ids = prefix_ids + filler_ids[: n - len(prefix_ids)]
        att = attentions(model, ids, device)               # по слоям: [heads, n, n]
        row = sum(a[:, -1].mean(0) for a in att) / len(att)  # среднее по слоям и головам, строка последнего токена
        share = float(row[instr].sum())
        sink = float(row[0])
        uniform = instr_len / len(ids)
        rows.append({"context_len": len(ids), "instruction_len": instr_len, "share_on_instruction": round(share, 5),
                     "sink_share": round(sink, 5), "uniform_share": round(uniform, 5)})
        log(f"  n={len(ids)}: доля на инструкцию {share:.4f} (равномерно было бы {uniform:.4f}), на сток {sink:.4f}")
        del att
    with open(out / "dilution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="режим CI: меньше слияний, короче контексты")
    ap.add_argument("--model", default=os.environ.get("LAB_MODEL", "Qwen/Qwen3-0.6B-Base"))
    ap.add_argument("--e5", default=os.environ.get("LAB_E5_MODEL", E5_MODEL), help="энкодер для поиска в части 3")
    ap.add_argument("--corpus", default=os.environ.get("LAB_CORPUS", str(HERE / "corpus.txt")))
    ap.add_argument("--out", default=str(HERE / "artifacts"), help="куда писать артефакты")
    ap.add_argument("--skip-model", action="store_true", help="только часть 1, без загрузки модели")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = load_raw(Path(args.corpus))
    text = normalize(raw)
    part_bpe(text, out, args.fast)
    if args.skip_model:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device()
    log(f"модель {args.model} на {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager").to(device).eval()

    part_tokenizer(text, tok, args.model, out)
    part_search(raw, tok, model, device, out, args.fast, args.e5)
    part_attention_map(model, tok, device, out)
    part_dilution(text, model, tok, device, out, args.fast)
    log("готово")


if __name__ == "__main__":
    main()
