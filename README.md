# Kafkaeske Steam-Reviews

Replikationspaket zur Arbeit *Wie wird Franz Kafka und das Kafkaeske in Nutzer*innen-Reviews auf Steam rezipiert?*  
Enthält: Korpus, Skripte, Beispiel-Ausgaben, Tabellen/Plots.

## Struktur
- `steam_reviews_kafka.csv` – Rohdaten (UTF-8; Spalten: doc_id,source,language,date,text)
- `analysis_kafka.py` – einfache Kollokationsanalyse
- `bar-,collocates,-kwic-...` – generierte Ergebnisse (CSV/PNG)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # (Win: .venv\Scripts\activate)
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
