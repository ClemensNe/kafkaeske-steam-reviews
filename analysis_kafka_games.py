#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replikationsskript für KWIC- und Kollokationsanalysen zu "Kafka" und "kafkaesk".

Autor*in: <Clemens Neuber>
Mit Formulierungshilfe & Code-Assistenz: ChatGPT (OpenAI, 2025).


Hinweis zur Transparenz:
  Dieses Skript dokumentiert, wo Zitate unverändert bleiben. KWIC-Export schreibt
  die Original-Satzsegmente ohne Normalisierung in die CSVs, inkl. Quelle & doc_id.
"""

import os
import json
import math
import argparse
import regex as re
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

# NLTK für Tokenisierung/Stopplisten
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# spaCy optional für Lemmata
try:
    import spacy
    SPACY_OK = True
except Exception:
    SPACY_OK = False

# -------------------------
# Konfiguration & Utilities
# -------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGETS = {
    "kafka": {
        "patterns": [
            r"\bKafka\b",       # Eigenname
            r"\bFranz\s+Kafka\b"
        ],
        "case_sensitive": True
    },
    "kafkaesk": {
        "patterns": [
            r"\bkafkaesk(e|er|en|es)?\b",  # dt.
            r"\bkafkaesque\b"              # en.
        ],
        "case_sensitive": False
    }
}

WORD_RE = re.compile(r"\p{L}[\p{L}\p{Mn}\-']*", re.IGNORECASE)

def build_stoplist(lang_codes: Iterable[str], remove_stopwords: bool, custom_path: str = None) -> Dict[str, set]:
    lang_map = {}
    if not remove_stopwords:
        return defaultdict(set)
    for lc in lang_codes:
        try:
            lang_map[lc] = set(stopwords.words('german')) if lc == 'de' else set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            lang_map[lc] = set(stopwords.words('german')) if lc == 'de' else set(stopwords.words('english'))
    # leichte Normalisierung
    for lc in lang_map:
        lang_map[lc] = {w.lower() for w in lang_map[lc]}
    # Custom ergänzen
    if custom_path and os.path.isfile(custom_path):
        with open(custom_path, 'r', encoding='utf-8') as f:
            custom = {line.strip().lower() for line in f if line.strip()}
        for lc in lang_map:
            lang_map[lc].update(custom)
    return lang_map

def get_spacy_pipes(use_lemma: bool) -> Dict[str, object]:
    pipes = {}
    if not use_lemma or not SPACY_OK:
        return pipes
    # Deutsch
    try:
        pipes['de'] = spacy.load("de_core_news_sm", disable=["ner", "parser", "textcat"])
    except Exception:
        pipes['de'] = None
    # Englisch
    try:
        pipes['en'] = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except Exception:
        pipes['en'] = None
    return pipes

def tokenize_sentences(text: str, lang: str) -> List[str]:
    # NLTK-Satzsegmentierung für de/en (Punkt/Abkürzungen robust genug i.d.R.)
    try:
        return sent_tokenize(text, language='german' if lang == 'de' else 'english')
    except LookupError:
        nltk.download('punkt')
        return sent_tokenize(text, language='german' if lang == 'de' else 'english')

def tokenize_words(text: str) -> List[str]:
    # robuste tokenisierung über Regex (Wortcharaktere, inkl. diakritika)
    return WORD_RE.findall(text)

def normalize_tokens(tokens: List[str], lang: str, remove_stop: bool, stoplist: Dict[str, set],
                     use_lemma: bool, nlp_map: Dict[str, object]) -> List[str]:
    # Lowercasing für Normalisierungen (aber: KWIC nutzt Original, hier nur für Stats)
    toks = [t.lower() for t in tokens]
    if use_lemma and nlp_map.get(lang):
        doc = nlp_map[lang](" ".join(toks))
        toks = [t.lemma_.lower() if t.lemma_ else t.text.lower() for t in doc]
    if remove_stop:
        toks = [t for t in toks if t not in stoplist[lang]]
    return toks

def matches_any(patterns: List[str], text: str, case_sensitive: bool) -> List[Tuple[int, int, str]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    spans = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags):
            spans.append((m.start(), m.end(), m.group(0)))
    spans.sort(key=lambda x: x[0])
    return spans

# -------------------------
# KWIC
# -------------------------

def kwic_for_doc(doc_text: str, lang: str, doc_id: str, source: str,
                 target_key: str, cfg: dict, window: int) -> List[dict]:
    """
    Gibt KWIC-Zeilen mit *Original* Kontext (keine Normalisierung!) zurück.
    """
    kwic_rows = []
    sents = tokenize_sentences(doc_text, lang)
    target = TARGETS[target_key]
    for sent in sents:
        # Finde Treffer per Regex im Originalsatz
        spans = matches_any(target["patterns"], sent, target["case_sensitive"])
        if not spans:
            continue
        # Tokenisiere *für Kontextfenster* (Wortweise)
        words = tokenize_words(sent)
        # baue Positionen auch auf Zeichenbasis -> Wortindex-Mapping
        # einfacher Ansatz: wir nutzen nur Wortfenster-Index über einfache Iteration
        # Ordne die gematchten Strings den Wortpositionen zu (heuristisch)
        # (Wenn das mal nicht exakt aligned ist, bleibt der Satz dennoch im Original erhalten.)
        for match in spans:
            # suche erste Wortposition, die den match.group im Wort enthält (heuristic)
            anchor_idxs = [i for i, w in enumerate(words) if re.search(re.escape(match[2]), w, re.IGNORECASE)]
            if not anchor_idxs:
                # Fallback: ganze Satz trotzdem als KWIC mit anchor_text
                kwic_rows.append({
                    "target": target_key,
                    "anchor": match[2],
                    "left": "",
                    "node": sent.strip(),
                    "right": "",
                    "doc_id": doc_id,
                    "source": source,
                    "language": lang
                })
                continue
            for aidx in anchor_idxs:
                left_start = max(0, aidx - window)
                right_end = min(len(words), aidx + window + 1)
                left = " ".join(words[left_start:aidx])
                node = words[aidx]
                right = " ".join(words[aidx+1:right_end])
                # Export mit Originalsatz zusätzlich (für Zitat-Genauigkeit)
                kwic_rows.append({
                    "target": target_key,
                    "anchor": match[2],
                    "left": left,
                    "node": node,
                    "right": right,
                    "sentence_original": sent.strip(),
                    "doc_id": doc_id,
                    "source": source,
                    "language": lang
                })
    return kwic_rows

# -------------------------
# Kollokationen & PMI
# -------------------------

def collect_window_counts(tokens: List[str], target_forms: List[str], window: int) -> Counter:
    """
    Zählt Kookkurrenzen (innerhalb ±window) für *normalisierte* Tokens.
    """
    counts = Counter()
    target_positions = [i for i, tok in enumerate(tokens) if tok in target_forms]
    for i in target_positions:
        l = max(0, i - window)
        r = min(len(tokens), i + window + 1)
        # exkludiere das Target selbst
        ctx = [t for j, t in enumerate(tokens[l:r]) if (l + j) != i]
        counts.update(ctx)
    return counts

def compute_collocations(df: pd.DataFrame, lang: str, target_key: str,
                         window: int, stoplist: Dict[str, set],
                         use_lemma: bool, nlp_map: Dict[str, object],
                         remove_stop: bool, pmi_min_freq: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gibt (collocations_df, freq_df).
    collocations_df enthält Token, f(token), f(target), f(token,target), PMI, NPMI, LLR (einfach),
    freq_df enthält reine Häufigkeiten im Fenster.
    """
    target = TARGETS[target_key]

    # Targetformen (normalisiert) vorbereiten:
    # Wir nehmen einfache Repräsentationen:
    target_forms_norm = set()
    # plausible Varianten: kafka, franz kafka, kafkaesk*, kafkaesque
    if target_key == "kafka":
        target_forms_norm.update(["kafka", "franz", "franz kafka"])
    else:
        target_forms_norm.update(["kafkaesk", "kafkaesque"])

    # Gesamtkorpus-Statistiken (für PMI)
    token_total = 0
    unigram_counts = Counter()
    window_counts = Counter()
    target_total = 0

    pipes = nlp_map

    for _, row in tqdm(df[df["language"] == lang].iterrows(), total=(df["language"] == lang).sum(), desc=f"Colloc {target_key}/{lang}"):
        text = row["text"]
        # Tokens normalisieren
        raw_tokens = tokenize_words(text)
        norm_tokens = normalize_tokens(raw_tokens, lang, remove_stop, stoplist, use_lemma, pipes)
        token_total += len(norm_tokens)
        unigram_counts.update(norm_tokens)

        # target-vorkommen approximieren: wir zählen tokens, die wie target_forms_norm aussehen
        # (einfach: wenn "kafka" in norm_tokens -> target_total++)
        doc_target_positions = [i for i, t in enumerate(norm_tokens) if t in target_forms_norm]
        target_total += len(doc_target_positions)

        window_counts.update(collect_window_counts(norm_tokens, list(target_forms_norm), window))

    # PMI/NPMI berechnen (nur ab Mindestfreq)
    rows = []
    N = token_total
    fT = max(target_total, 1)
    for tok, f_t_given_T in window_counts.items():
        if f_t_given_T < pmi_min_freq:
            continue
        f_t = unigram_counts.get(tok, 0)
        if f_t == 0:
            continue
        # Joint ~ Annäherung: K(T,t) = f_t_given_T
        # P(T,t) ≈ f_t_given_T / N
        # P(T) ≈ fT / N
        # P(t) ≈ f_t / N
        p_joint = f_t_given_T / N
        p_t = f_t / N
        p_T = fT / N
        # Schutz gegen log(0)
        if p_joint <= 0 or p_t <= 0 or p_T <= 0:
            continue
        pmi = math.log(p_joint / (p_t * p_T), 2)
        # Normalized PMI
        npmi = pmi / (-math.log(p_joint, 2))
        # einfacher LLR-Proxy (nicht die volle Dunning-LLR, aber informativ)
        # Hier: log-likelihood ratio ~ 2 * f_joint * log( (f_joint * N) / (f_t * f_T) )
        # Achtung: heuristisch
        llr = 2.0 * f_t_given_T * math.log((f_t_given_T * N) / max(1, (f_t * fT)), 2)

        rows.append({
            "token": tok,
            "freq_in_window": f_t_given_T,
            "freq_token_corpus": f_t,
            "freq_target": fT,
            "PMI": pmi,
            "NPMI": npmi,
            "LLR_proxy": llr
        })

    colloc_df = pd.DataFrame(rows).sort_values(["PMI", "freq_in_window"], ascending=[False, False]).reset_index(drop=True)
    freq_df = pd.DataFrame(window_counts.items(), columns=["token", "freq_in_window"]) \
                .sort_values("freq_in_window", ascending=False).reset_index(drop=True)

    return colloc_df, freq_df

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/reviews.csv")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--window", type=int, default=5, help="Kontextfenster ±N Wörter")
    parser.add_argument("--pmi_min_freq", type=int, default=5, help="Min. Fensterfreq für PMI-Zeile")
    parser.add_argument("--lemmatize", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--remove_stopwords", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--custom_stopwords", type=str, default="", help="Pfad zu optionaler Stoppliste")
    args = parser.parse_args()

    use_lemma = (args.lemmatize.lower() == "true")
    remove_stop = (args.remove_stopwords.lower() == "true")

    os.makedirs(args.out_dir, exist_ok=True)

    # Lade Daten
    df = pd.read_csv(args.data_path)
    needed = {"doc_id", "source", "language", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV fehlt Spalten: {missing}")

    # Nur de/en (falls vorhanden)
    df = df[df["language"].isin(["de", "en"])].copy()
    df["text"] = df["text"].fillna("").astype(str)

    # Stoplisten & spaCy
    stop_map = build_stoplist(["de", "en"], remove_stopwords=remove_stop, custom_path=args.custom_stopwords)
    nlp_map = get_spacy_pipes(use_lemma=use_lemma)

    # KWIC je Target
    all_kwic = []
    for target_key in ["kafka", "kafkaesk"]:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"KWIC {target_key}"):
            kw = kwic_for_doc(
                doc_text=row["text"],
                lang=row["language"],
                doc_id=str(row["doc_id"]),
                source=str(row["source"]),
                target_key=target_key,
                cfg=TARGETS[target_key],
                window=args.window
            )
            all_kwic.extend(kw)

        kwic_df = pd.DataFrame(all_kwic)
        # Nur Zeilen für das aktuelle Target filtern (weil all_kwic akkumuliert)
        kwic_df = kwic_df[kwic_df["target"] == target_key].copy()
        out_k = os.path.join(args.out_dir, f"kwic_{target_key}.csv")
        kwic_df.to_csv(out_k, index=False, encoding="utf-8")

    # Kollokationen je Sprache × Target
    for lang in ["de", "en"]:
        for target_key in ["kafka", "kafkaesk"]:
            colloc_df, freq_df = compute_collocations(
                df=df,
                lang=lang,
                target_key=target_key,
                window=args.window,
                stoplist=stop_map,
                use_lemma=use_lemma,
                nlp_map=nlp_map,
                remove_stop=remove_stop,
                pmi_min_freq=args.pmi_min_freq
            )
            colloc_df.to_csv(os.path.join(args.out_dir, f"collocations_{target_key}_{lang}.csv"),
                             index=False, encoding="utf-8")
            freq_df.to_csv(os.path.join(args.out_dir, f"freq_windows_{target_key}_{lang}.csv"),
                           index=False, encoding="utf-8")

    # Metadaten der Lauf speichern
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "params": {
            "window": args.window,
            "pmi_min_freq": args.pmi_min_freq,
            "lemmatize": use_lemma,
            "remove_stopwords": remove_stop,
            "custom_stopwords": args.custom_stopwords
        },
        "data_path": args.data_path,
        "out_dir": args.out_dir,
        "notes": [
            "KWIC exportiert Original-Satzsegmente (sentence_original) für exaktes Zitieren.",
            "Kollokationen basieren auf normalisierten Tokens (optional Lemma/Stopwortentfernung).",
            "PMI/NPMI sind heuristisch aus Fensterfrequenzen geschätzt; für strikte Inferenz ggf. LLR nach Dunning separat implementieren."
        ],
        "provenance": {
            "script": "scripts/analysis_kafka_games.py",
            "assistant": "ChatGPT (OpenAI, 2025) – Formulierungshilfe & Code-Vorlage",
            "citation": "OpenAI (2025). ChatGPT (Version GPT-5) [Large language model]. https://chat.openai.com/"
        }
    }
    with open(os.path.join(args.out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Fertig. Ergebnisse unter: {args.out_dir}")

if __name__ == "__main__":
    main()
