# AI Usage Log — Assignment 2

**Student:** Thomas Schlaerth
**AI tier:** 2
**AI tool used:** Claude (Anthropic)

---

## What I used AI for (allowed under Tier 2)

| Task | Why I asked AI | Outcome |
|---|---|---|
| Scaffolding the full notebook (imports, EDA structure, cleaning function, vectorizer comparison loop, training loop) | I wanted a clean, well-commented skeleton I could run top-to-bottom instead of building each cell from scratch | Got a runnable baseline that I then customized for my environment |
| Adapting the dataset loader to Google Colab via `kagglehub` | The original scaffold assumed a local CSV at `../data/`; I work in Colab and prefer auto-download | Swapped the `pd.read_csv` cell for a `kagglehub.dataset_download(...)` flow |
| Rewriting Step 6.4 (20-row manual review) into a two-cell export → fill-in-Excel → read-back workflow | I had used the same pattern in my ESG assignment and wanted to reuse the Excel-based workflow here | Got the export-to-xlsx and the compare-CSV cells; renamed `My Label` → `Manual Prediction` to match my Excel column naming |
| Generating the 20 custom inference examples (Step 7.1) | The assignment explicitly permits this. I asked for a balanced mix: 10 easy, 5 tricky (sarcasm / mixed / negation / flipped expectation / comparison), 5 out-of-domain (video game / music / book / TV / restaurant) | Got 20 examples spanning all three buckets that I then used as-is for inference |
| Idea-generation for Step 6.3 (important class identification) | I wanted to see the *space* of defensible business scenarios before committing to one | The AI gave me a menu of five deployment framings (recommender, PR monitoring, aggregator, moderation queue, hidden-gems surfacer). I picked the recommender framing on my own and wrote the reasoning in my own words |

