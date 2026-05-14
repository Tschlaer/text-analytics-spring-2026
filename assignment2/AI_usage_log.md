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

## What I did **without** AI (Tier 2 restrictions)

- **Important-class identification (§6.3)** — I read the menu of options, picked the recommender scenario because it most clearly justifies focusing on positive-class precision, and wrote the full paragraph myself.
- **5-criteria comparison and deployment recommendation (§6.5)** — I filled the comparison table from my own notebook outputs and wrote the recommendation in my own words.
- **Manual review error analysis (§6.4)** — I read each of the 20 reviews myself in Excel, labeled them, and noticed the pattern that long synopsis-style reviews and reviews making real-world comparisons were the hardest cases.
- **Reflection on the 20 custom examples (§7.3)** — All five answers in `reflection.md` are my own writing based on what I saw the model do.

---

## Prompt evolution — A1 → A2

Comparing to Assignment 1, I noticed my prompts became less open-ended and more structured.

### Early-style prompt (vague, like A1)

> "Help me complete the assignment."

The AI did produce a useful skeleton, but I would not have known what to do with it without already understanding the pipeline. In A1 this kind of prompt was my default — and I spent a lot of time fixing code that did not fit my actual data.

### Mid-assignment prompt (more specific)

> "Give me some ideas for step 6.3 on the assignment given the code output."

This one was scoped to a *specific section* and asked for *ideas, not a finished answer.* The response gave me five framings to compare instead of one canned recommendation, which kept the decision in my hands.

### Late-assignment prompt (context-rich, constrained)

> "Remake the code for 6.4 so it aligns with this code [pasted my ESG-assignment two-cell export-then-compare pattern]."

This was the pattern that worked best: state what I already have, state what I want, paste the reference style. The AI returned exactly the rewrite I needed (export to xlsx, manual fill, read-back comparison) and only changed the parts that had to change for the IMDB context (string labels instead of `LabelEncoder` classes; column name change from "My Label" to "Manual Prediction").

### Lesson learned

The prompts that wasted the least time gave the AI three things: **the existing context**, **the desired output**, and **the constraint** (e.g., "match this pattern" or "do not change the rest"). Vague "help me with X" prompts produced generic code that needed more editing than writing from scratch.

---

## When my own approach beat AI's suggestion

The AI's original scaffold had me read the manual-review CSV from `../figures/manual_review.csv` — but I was working in Google Colab, where I had downloaded the filled-in CSV back into `/content/`. I changed the read path to `/content/manual_review.csv` and the cell worked on the first run. The AI's path assumed a local file-system layout that did not match my actual environment.

A second case: when the AI rewrote §6.4, it used `'My Label'` as the column name for the manual annotation. I had been working with `'Manual Prediction'` in my ESG assignment and preferred to stay consistent, so I edited the column name in both the export cell and the compare cell. Small change, but the AI did not know my preferred convention until I told it.

---

## Honest accounting

- I did **not** use AI to write §6.3 (important class identification), §6.4 (manual-review error patterns), §6.5 (5-criteria recommendation), or `reflection.md`. Those are my own writing.
- I did use AI to write the implementation code in the notebook and to generate the 20 custom inference examples in §7.1.
- The numbers reported in the notebook and in `README.md` are the actual outputs from running the code in Colab.
