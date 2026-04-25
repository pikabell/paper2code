# Installation Guide — paper2code

This fork adds `--pdf` support so you can convert **any PDF paper** to code — not just arXiv papers.

---

## Prerequisites

- [npx](https://www.npmjs.com/) (comes with Node.js)
- Python 3.8+
- `pip` for dependency installation

---

## Step 1: Install the skill

```bash
npx skills add pikabell/paper2code@paper2code -g -y
```

This installs globally. To install per-project, omit `-g`.

---

## Step 2: Verify

Run your agent (e.g., `claude`) and check the skill is available:

```
/help
```

Look for `paper2code` in the skill list.

---

## Usage

### From arXiv

```bash
/paper2code https://arxiv.org/abs/1706.03762
```

### From any PDF (local file)

```bash
/paper2code --pdf /path/to/paper.pdf \
  --title "Paper Title Here" \
  --authors "First Author, Second Author" \
  --abstract "Brief paper abstract..."
```

**Required arguments for local PDFs:**
- `--pdf` — path to the PDF file
- `--title` — the paper's title

**Optional:**
- `--authors` — comma-separated author names
- `--abstract` — paper abstract

---

## What you get

The skill generates a `{paper_slug}/` directory with:
- `src/model.py` — architecture with § section citations
- `src/loss.py` — loss functions with equation references
- `src/train.py` — training loop (if in scope)
- `configs/base.yaml` — all hyperparameters, each cited or flagged `[UNSPECIFIED]`
- `REPRODUCTION_NOTES.md` — full ambiguity audit
- `notebooks/walkthrough.ipynb` — runnable sanity checks

---

## Updating the skill

```bash
npx skills update pikabell/paper2code@paper2code
```

---

## Uninstall

```bash
npx skills remove pikabell/paper2code@paper2code
```