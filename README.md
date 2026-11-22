# Persona Vectors auf Qwen3-4B

Projekt zur Vorlesung Explainable AI: Übertragung der Befunde von Chen et al. (2025), *Persona Vectors*, für den Trait Sycophancy auf ein neueres Modell.

| | |
|---|---|
| Name | [Name eintragen] |
| Matrikelnummer | [Matrikelnummer eintragen] |
| E-Mail | [E-Mail eintragen] |
| GitHub | https://github.com/Loitt2/Persona-Vector |

## Forschungsfrage

Lassen sich zwei Befunde von Chen et al. (2025, [arXiv:2507.21509](https://arxiv.org/abs/2507.21509)) für den Trait Sycophancy auf Qwen3-4B übertragen?

1. **Steuerbarkeit:** Erhöht Activation Steering mit dem Persona-Vektor die Schmeichelei des Modells?
2. **Vorhersagekraft:** Sagt die Projektion der Aktivierung auf den Vektor vor der Antwort vorher, wie schmeichlerisch die Antwort wird?

Einordnung: Post-hoc, Mechanistic Interpretability, Decoder-Transformer, Difference-in-Means, Activation Steering, Projection Monitoring.

## Ergebnisse und Dokumentation

Die vollständige Dokumentation mit Methoden, Plots, Ergebnissen und Grenzen steht im Notebook
[`notebooks/persona_vectors_qwen3.ipynb`](notebooks/persona_vectors_qwen3.ipynb). Es ist bereits ausgeführt gespeichert.

## Daten und Modelle

- **Datensatz:** Es gibt keinen klassischen Datensatz. Verwendet werden die Trait-Artefakte der Paper-Autoren, vor allem für `sycophantic`
  und für die Trait-Landkarte auch die sechs übrigen Traits (System-Prompt-Paare, Fragen, Judge-Rubrik) aus https://github.com/safety-research/persona_vectors,
  Ordner `data_generation/trait_data_extract/` und `data_generation/trait_data_eval/`. Das Repo ist als Git-Submodul unter
  `external/persona_vectors` eingebunden.
- **Zielmodell:** [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), wird beim ersten Lauf von Hugging Face geladen (rund 8 GB).
- **Judge:** Claude Opus 5.5 über die Anthropic-API, zur Kontrolle zusätzlich Claude Haiku 5.5.
- **Eigene Ergebnisse:** Alle erzeugten Antworten, Scores und die extrahierten Vektoren liegen in `results/` (rund 25 MB, überwiegend Textdateien; gepackt deutlich kleiner).

## Setup

Voraussetzung: Python 3.12.

```bash
git clone --recursive https://github.com/Loitt2/Persona-Vector
cd Persona-Vector
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

Die `requirements.txt` enthält PyTorch mit CUDA 13.0. Ohne passende Grafikkarte genügt für das Notebook eine CPU-Version von PyTorch.

### Nur die Ergebnisse nachvollziehen (ohne GPU, ohne API-Key)

```bash
jupyter notebook notebooks/persona_vectors_qwen3.ipynb
```

Das Notebook erklärt die Grundlagen, nennt zu jedem Experiment das zugehörige Skript und erzeugt alle Plots aus den Dateien in `results/`.
Es zeigt die Hauptexperimente zu Sycophancy, den Kritik-Versuch, die Abweichungen vom Paper und die Versuche mit bekannten Figuren.

### Experimente neu ausführen (GPU mit rund 12 GB und Anthropic-API-Key nötig)

Im Projektordner eine Datei `.env` mit der Zeile `ANTHROPIC_API_KEY=...` anlegen. Dann:

```bash
python scripts/00_smoke_test.py     # Modell laden, Aktivierungen und Steering-Hook prüfen
python scripts/01_judge_test.py     # Judge an drei handgeschriebenen Antworten prüfen
python scripts/start.py             # Stand aller Schritte anzeigen, nichts ausführen
python scripts/start.py --run       # fehlende Schritte der Reihe nach ausführen
python scripts/zusammenfassung.py   # Kernzahlen aller Experimente drucken (ohne GPU und Key)
python scripts/demo.py              # live ausprobieren: Frage eintippen, Steering-Stärke einstellen
```

`start.py` ruft die einzelnen Experiment-Skripte in der richtigen Reihenfolge auf; sie lassen sich auch einzeln starten (siehe Tabelle unten).
Ein Schritt gilt als erledigt, wenn seine Ergebnisdatei in `results/` liegt.

Die Generierung nutzt Sampling (Temperatur 1, top-k 20); ein neuer Lauf liefert deshalb ähnliche, aber nicht identische Zahlen.
Die Judge-Aufrufe der Experimente 00 bis 18 haben zusammen rund 35 US-Dollar gekostet; die Nachträge (Skripte 19 bis 22) kommen dazu.

## Aufbau des Repos

| Pfad | Inhalt |
|---|---|
| `notebooks/persona_vectors_qwen3.ipynb` | Dokumentation, Auswertung und Ergebnisse |
| `src/pv_common.py` | Modell laden, Prompts bauen, Antworten erzeugen |
| `src/pv_judge.py` | LLM-Judge |
| `src/pv_steering.py` | Generierung mit Eingriff in die Aktivierungen |
| `src/pv_data.py` | Selbst geschriebenes Material: abgestufte System-Prompts, neue Fragen, Figuren, eigene Rubriken |
| `src/pv_characters.py` | Bausteine für die Figuren-Versuche und für Eingriffe, die über Layer vergleichbar sind |
| `results/` | Antworten und Scores als CSV, Vektoren als `.pt` |
| `results/figures/` | Die wichtigsten Abbildungen des Notebooks als PNG |
| `external/persona_vectors` | Repo der Paper-Autoren (Submodul) |

Die Experiment-Skripte in `scripts/`:

| Skript | Experiment |
|---|---|
| `02_extract.py` | Persona-Vektor extrahieren (`generate`, `judge`, `vector`; `--trait` für andere Traits) |
| `03_steering.py` | Layer-Suche (`layers`) und Steering-Kurve (`curve`) |
| `04_monitoring.py` | Projektion vor der Antwort gegen das Verhalten |
| `05_controls.py` | Kontrollen: Zufallsvektor, negatives Vorzeichen |
| `06_logit_lens.py` | Logit Lens auf den Vektor |
| `07_monitoring_layers.py` | Monitoring über alle Layer, mit zwei Vektoren |
| `08_second_judge.py` | Zweiter Judge (Claude Haiku 5.5) |
| `09_new_formats.py` | Neue Fragenformate: falsche Behauptungen und schwache Arbeit |
| `10_length_control.py` | Steering mit einem Vektor aus den ersten 50 Antwort-Tokens |
| `11_trait_map.py` | Ähnlichkeit der Vektoren von sieben Traits |
| `12_token_heatmap.py` | Projektion Token für Token |
| `13_other_traits.py` | Steering und Monitoring für evil und hallucination |
| `14_critique_monitoring.py` | Faire gegen abwertende Kritik: Projektion vor der Antwort |
| `15_characters.py` | Persönlichkeitsprofile zehn bekannter Figuren |
| `16_character_steering.py` | Darth Vader ohne „böse" im Vergleich mit Anakin Skywalker |
| `17_layer_search.py` | Faire Layer-Suche für die drei Haupt-Traits |
| `18_anakin_layers.py` | Der Vader-Anakin-Versuch über alle Layer |
| `19_evil_retest.py` | Nachtest: evil mit feineren Stärken und auf Layer 20 |
| `20_example_no_prompt.py` | Beispielantworten ohne System-Prompt für das Notebook |
| `21_more_controls.py` | Nachträge: Layer 18 mit höheren Stärken, 19 weitere Zufallsvektoren |
| `22_vader_to_anakin.py` | Darth Vader gezielt in Richtung Anakin steuern; welche Traits die beiden unterscheiden |
| `23_vader_trait_mix.py` | Darth Vader mit einer Mischung aus vier Trait-Vektoren steuern |
| `24_role_vector.py` | „Das Modell spielt eine Rolle" als eigene Richtung messen und aus den Trait-Vektoren entfernen |
| `25_evil_without_role.py` | Evil-Vektor ohne Rollen-Anteil bei Darth Vader und beim normalen Assistenten addieren |
| `26_villains_evil.py` | Dasselbe für Voldemort, Joker und Ned Flanders |
| `27_mix_all_characters.py` | Die Trait-Mischung bei allen zehn Figuren addieren |
| `28_gym_demo.py` | Eine Alltagsfrage ohne Eingriff und mit drei Persona-Vektoren, als Beispiel für den Vortrag |

Ab Skript 13 bewertet Claude Haiku 5.5 statt Opus 5.5; beide stimmen laut `08_second_judge.py` zu r = 0,93 überein.

## Was aus dem Original-Repo stammt und was neu ist

- **Übernommen:** die Trait-Artefakte für Sycophancy, die Kohärenz-Rubrik und der Steering-Hook (`activation_steer.py`).
- **Neu geschrieben:** Generierung mit Hugging Face Transformers statt vLLM (das Original läuft nicht unter Windows), der Judge für die Anthropic-API,
  alle Experiment-Skripte, die abgestuften System-Prompts fürs Monitoring und für den Kritik-Versuch, 20 Fragen in neuen Formaten,
  die Figuren mit ihren Alltagsfragen, die Rubriken „Deutlichkeit der Korrektur" und „Wer spricht?", die Kontrollen, die Erweiterungen und die Auswertung.

## Quellen

1. Chen, R., Arditi, A., Sleight, H., Evans, O., Lindsey, J. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
2. Turner, A. M. et al. (2023). *Steering Language Models With Activation Engineering.* arXiv:2308.10248.
3. Rimsky, N. et al. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. arXiv:2312.06681.
4. Sharma, M. et al. (2023). *Towards Understanding Sycophancy in Language Models.* arXiv:2310.13548.
5. Qwen Team (2025). *Qwen3 Technical Report.* arXiv:2505.09388.
6. nostalgebraist (2020). *interpreting GPT: the logit lens.* LessWrong.

## Einsatz von KI-Werkzeugen

- Umsetzung des Codes
- Die Texte mit KI-Unterstützung formuliert und von mir bearbeitet.
- Einarbeiten in das Thema

Alle Ergebnisse habe ich nachvollzogen und kann sie erläutern.
