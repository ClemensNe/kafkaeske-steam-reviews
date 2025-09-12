
import pandas as pd, re, os, math
from collections import Counter, defaultdict

def clean_markups(s: str) -> str:
    s = re.sub(r"\[/?[^\]]+\]", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s

EN_STOP = set("a an and are as at be by for from has he in is it its of on that the to was were will with i you your we they this those these there here or if but so than then them our about into over under very not no yes do did done does can could should would might must may up out more most less least also just only even much many such other another".split())
DE_STOP = set("der die das ein eine einer eines einem einen und ist sind war waren wird wurden worden werden ich du er sie es wir ihr man mir mich dich ihn ihr ihnen uns euch mein meine meiner meinem meinen dein deine deiner deinem deinen sein seine seiner seinem seinen ihr ihre ihrer ihrem ihren von im in ins auf aus bei mit nach vor hinter unter über zwischen wegen ohne durch nicht kein keine keiner keinem keinen auch noch nur schon aber oder denn sondern so wie als mehr sehr bis zum zur vom am".split())
STOP = EN_STOP | DE_STOP

def simple_tokenize(text: str):
    text = re.sub(r"[^\w\s']", " ", str(text), flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.split()

def rule_based_lemma(token: str) -> str:
    t = token.lower()
    if t.endswith("'s"): t = t[:-2]
    if len(t) > 4 and t.endswith("ies"): t = t[:-3] + "y"
    elif len(t) > 3 and t.endswith("sses"): pass
    elif len(t) > 3 and t.endswith("xes"): t = t[:-2]
    elif len(t) > 3 and t.endswith("ches"): t = t[:-2]
    elif len(t) > 3 and t.endswith("shes"): t = t[:-2]
    elif len(t) > 3 and t.endswith("es"): t = t[:-2]
    elif len(t) > 2 and t.endswith("s") and not t.endswith("ss"): t = t[:-1]
    if t.endswith("en") and len(t) > 4: t = t[:-2]
    if t.endswith("e") and len(t) > 4: t = t[:-1]
    return t

def collocates(texts, targets, window=5):
    tokens_all, target_positions = [], []
    for doc_id, s in enumerate(texts):
        toks = simple_tokenize(s)
        tokens_all.append(toks)
        for i, tok in enumerate(toks):
            if tok in targets:
                target_positions.append((doc_id, i))
    vocab_counts = Counter(); joint_counts = Counter(); target_counts = Counter(); N = 0
    for toks in tokens_all:
        for tok in toks:
            lemma = rule_based_lemma(tok)
            vocab_counts[lemma] += 1; N += 1
    for doc_id, i in target_positions:
        toks = tokens_all[doc_id]
        center_tok = rule_based_lemma(toks[i])
        target_counts[center_tok] += 1
        start = max(0, i-window); end = min(len(toks), i+window+1)
        for j in range(start, end):
            if j == i: continue
            w = rule_based_lemma(toks[j])
            if w in STOP: continue
            if w in targets: continue
            joint_counts[(center_tok, w)] += 1
    coll = defaultdict(lambda: {"freq":0, "pmi":0.0, "t":0.0, "n_y":0})
    total_target = sum(target_counts.values())
    for (t, w), n_xy in joint_counts.items():
        n_x = total_target; n_y = vocab_counts[w]
        pmi_val = math.log2(((n_xy + 1) * N) / ((n_x + 1) * (n_y + 1)))
        exp = (n_x * n_y) / N if N>0 else 0.0
        t_val = (n_xy - exp) / math.sqrt(n_xy) if n_xy>0 else 0.0
        coll[w]["freq"] += n_xy; coll[w]["n_y"] = n_y; coll[w]["pmi"] = max(coll[w]["pmi"], pmi_val); coll[w]["t"] += t_val
    out = [(w, d["freq"], d["pmi"], d["t"], d["n_y"]) for w,d in coll.items()]
    return pd.DataFrame(out, columns=["collocate","window_freq","pmi","t_score","global_freq"]).sort_values(by=["window_freq","pmi"], ascending=False)

def main(in_csv="/mnt/data/steam_reviews_kafka (1).csv", out_dir="/mnt/data/kafka_outputs"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    df["clean"] = df["review"].apply(clean_markups)
    df_en = df[df["review_language"].str.lower().eq("english")]
    df_de = df[df["review_language"].str.lower().eq("german")]
    coll_kafka = collocates(df["clean"].tolist(), {"kafka"}, window=5)
    coll_kafka.to_csv(os.path.join(out_dir,"collocates_kafka_all.csv"), index=False)
    coll_kafkaesque = collocates(df_en["clean"].tolist(), {"kafkaesque"}, window=5)
    coll_kafkaesque.to_csv(os.path.join(out_dir,"collocates_kafkaesque_all.csv"), index=False)
    coll_kafkaesk = collocates(df_de["clean"].tolist(), {"kafkaesk"}, window=5)
    coll_kafkaesk.to_csv(os.path.join(out_dir,"collocates_kafkaesk_all.csv"), index=False)

if __name__ == "__main__":
    main()
