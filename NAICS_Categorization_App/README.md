# NAICS Classifier

A desktop app that automatically assigns NAICS (North American Industry Classification System) codes to procurement line items.

Manually assigning NAICS codes to thousands of purchase orders is slow and inconsistent. This tool automates it: give it a spreadsheet with item descriptions (and optionally supplier names), and it predicts the most likely NAICS code for each row — along with a confidence score and up to your chosen number of alternatives.

**How it works:** Descriptions are sent to OpenAI's `text-embedding-3-large` model to produce numeric embeddings, 
which are fed into a locally-trained XGBoost classifier. Only description and supplier text will be sent to OpenAI 
servers — everything else stays local.

The tool has two phases:
1. **Train** — provide labeled data (items with correct NAICS codes) so the model learns the patterns.
2. **Predict** — provide new, unlabeled data and the trained model assigns codes automatically.

---

## Before you start — OpenAI API setup

This app requires an OpenAI API key and a small amount of prepaid credits. **Read [OPENAI_API_SETUP.md](OPENAI_API_SETUP.md) first** — it walks through creating an account, generating a key, restricting its permissions, and turning off auto-recharge to protect against unexpected charges.

For pricing, go to https://developers.openai.com/api/docs/pricing, find **Specialized models**, and click **View more** to find `text-embedding-3-large` (hidden by default).

---

## First-time setup

### Mac

**1. Install Python 3**

Download and install Python 3 from https://www.python.org/downloads/ and follow the prompts.

**2. Install Homebrew, then libomp**

Homebrew is a Mac package manager required for XGBoost. Follow the installation instructions at https://brew.sh/

Once Homebrew is installed, open Terminal (press **Command + Space**, type `Terminal`, press Enter) and run:

```bash
brew install libomp
```

**3. Navigate to the app folder**

The `cd` command tells Terminal which folder to work in. You need to point it to wherever you saved the `NAICS_Categorization_App` folder on your computer.

The easiest way — no typing required:

1. Type `cd ` in Terminal (c, d, space). **Do not press Enter yet.**
2. Open Finder and locate your `NAICS_Categorization_App` folder.
3. Drag that folder from Finder directly onto the Terminal window. The full path will appear after `cd `.
4. Now press Enter.

It will look something like this (the exact path depends on where you saved it), note that there is a space between cd 
and the folder path:

```
cd /Users/yourname/Desktop/NAICS_Categorization_App
```

**4. Install packages**

```bash
python3 -m pip install -r requirements.txt
```

This only needs to be done once.

---

### Windows

**1. Install Python 3**

Download and install Python 3 from https://www.python.org/downloads/. On the first screen, check **"Add Python to PATH"** before clicking Install.

**2. Navigate to the app folder**

The `cd` command tells Command Prompt which folder to work in. You need to point it to wherever you saved the `NAICS_Categorization_App` folder.

The easiest way:

1. Open File Explorer and navigate to your `NAICS_Categorization_App` folder.
2. Click the address bar at the top of the File Explorer window — it will show the full folder path and highlight it.
3. Copy that path (Ctrl + C).
4. Open Command Prompt: press the **Windows key**, type `cmd`, press Enter.
5. Type `cd ` (c, d, space), then paste the path you copied (right-click → Paste, or Ctrl + V). Press Enter.

It will look something like this:

```
cd C:\Users\yourname\Desktop\NAICS_Categorization_App
```

If the path contains spaces, wrap the whole thing in quotes:

```
cd "C:\Users\yourname\My Documents\NAICS_Categorization_App"
```

**3. Install packages**

```
python -m pip install -r requirements.txt
```

This only needs to be done once.

---

## Running the app

Every time you want to launch the app, you need to open Terminal (Mac) or Command Prompt (Windows) and navigate to the app folder first — Terminal does not remember where you were last time.

**Mac:**
```bash
cd /Users/yourname/Desktop/NAICS_Categorization_App
python3 app.py
```

**Windows:**
```
cd C:\Users\yourname\Desktop\NAICS_Categorization_App
python app.py
```

Replace the path with wherever you saved the folder (same path you used during setup). You do not need to reinstall packages — just the `cd` and the launch command.

---

## Train Tab

Build a model from labeled data (items that already have correct NAICS codes).

| Field | What to enter |
|---|---|
| **Output directory** | Where to save the `.naics_model` file |
| **Model name** | A label for your model (e.g. `FY24_Procurement`) |
| **Description column** *(required)* | Column with item descriptions |
| **Supplier column** *(optional)* | Vendor/supplier name; check **Not in my data** if absent |
| **NAICS/EEIO column** *(required)* | Column with correct NAICS codes — what the model learns |
| **NAICS description column** *(optional)* | Human-readable label for each code; check **Not in my data** if absent |

**Category-Based Training** *(optional)*: when enabled, one specialized model is trained per spending category plus a general model on all rows — all packed into a single `.naics_model` file. At prediction time, each row is routed to its matching category model; unrecognized categories fall back to the general model. Useful when your data spans very different purchase types (IT hardware, office supplies, lab equipment, etc.). Each category needs at least 2 rows and 2 unique NAICS codes to be included.

**API Key**: enter your `sk-...` key. Click **Test** before starting a long run. Check **Save API key to config file** only if this is your own personal computer — never save the key on a shared machine, and never share the `config.json` file with others. For guidance on key restrictions and rate limits, see [OPENAI_API_SETUP.md](OPENAI_API_SETUP.md).

**Hyperparameters** (defaults work well for most datasets):

| Field | Default | What it controls |
|---|---|---|
| **Embedding batch size** | 500 | Items sent to OpenAI per API call |
| **ML Model (XGBoost) depth** | 100 | How deep decision trees grow — higher can learn more complex patterns but risks overfitting |
| **Training rounds** | 50 | Boosting iterations — more rounds generally means better accuracy |

---

## Predict Tab

Apply a trained model to new, unlabeled data.

Browse to a `.naics_model` file, load your input CSV or Excel file, map the description and supplier columns, enter your API key, and click Run.

Results are saved as a CSV in the same folder as your input file by default. Output columns added per prediction slot (up to your chosen **Top-K**):

| Column | Contents |
|---|---|
| `ML_pred1_NAICS` | Top predicted NAICS code |
| `ML_pred1_confidence` | Confidence score (0–1) |
| `ML_pred1_description` | Human-readable label for the code |
| `ML_pred2_NAICS`, `ML_pred2_confidence`, `ML_pred2_description` | Second prediction, and so on |
| `ML_model_used` | *(multi-model bundles only)* which model handled this row |

---

## Embedding Checkpoints

Getting embeddings from OpenAI is the only step that costs money. A checkpoint saves the result so you can reuse it on future runs without paying again.

- **Save checkpoint**: after embedding completes, the app writes a `.naics_embed` file.
- **Load checkpoint**: skips the embedding step entirely and reads a previously saved file. No API call or key required.

**When to use**: if you want to experiment with different training settings on the same dataset, save the checkpoint on the first run and load it each time you retrain. For a brand-new dataset you always need to embed, but you can save that result too. For prediction, the same logic applies — save if you might re-run the model on the same data.

---

## Billing

- **Start with a small test file** (10–20 rows) before running on your full dataset.
- **Keep auto-recharge OFF** and load only the minimum credits you need. See [OPENAI_API_SETUP.md](OPENAI_API_SETUP.md) for how to configure this.
- Monitor live usage at https://platform.openai.com/settings/organization/usage
