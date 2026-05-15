# NAICS Classifier

A desktop app that automatically assigns NAICS (North American Industry Classification System) codes to procurement line items.

Manually assigning NAICS codes to thousands of purchase orders is slow and inconsistent. This tool automates this process, where one can used a pre-trained LLM from USC's team or train a new LLM and then upload a spreadsheet with item descriptions (and optionally supplier names) in order to predict the most likely NAICS code for each row (based on the text from the item category, item label/description and/or supplier). This tool will output NAICS codes and descriptions along with a confidence score for each prediction and up to your chosen number of alternative predictions.

**How it works:** Descriptions are sent to OpenAI's `text-embedding-3-large` model to produce numeric embeddings, 
which are fed into a locally-trained XGBoost classifier. Only the provided item description (label) and supplier text will be sent to OpenAI 
servers — everything else stays on your local computer.

The tool has two phases:
1. **Train** — provide labeled data (items with correct NAICS codes) so the model learns the patterns. 
2. **Predict** — provide new, unlabeled data and the trained model assigns codes automatically.

The first phase (Train) is optional in the event that you do not want to use the LLM based on USC's training dataset (FY22-FY24 procurement data with primarily manually assigned NAICS codes). You can skip to the second phase (Predict) and upload [USC's NAICS prediction model](https://drive.google.com/file/d/1dGSdRwLU5hAGIQYfVg7NqQSVKA5GweOf/view?usp=share_link) instead of training your own model if you would like. 

**Important Notes and Tips!**

We have several recommendations: 
1. Pre-process your training dataset (for training a new model) and/or your test datasets (for predicting NAICS codes) to decrease the processing time.
- [Rscript for pre-processing data before training model or predicting output](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/edit/main/NAICS_Categorization_App/README.md#:~:text=Processing_prior_to_LLM.R)
- [Python script (only pre-proceses data before training the model)](https://github.com/USC-Office-of-Sustainability/Scope-3-Emissions/blob/main/NAICS_Categorization_App/prepare_NAICS_data.py)
2. We **STRONGLY** recommend that you check the NAICS code output results for your data, especially for where the LLM (tool) outputs a lower confidence level (anything < 50%). You can also cross-check the model predictions against an unseen dataset that you have also previously manually checked and then do a 'True-False' match to see which ones are wrong. Please keep in mind that sometimes manual assignments can be wrong due to human error as well. 

Once you manually check your output data and correct each NAICS code that needs to be corrected, you can always create a larger training dataset to train a new model before you run 'predict' on a new unseen dataset. This is what we did at USC so that we could improve our model output accuracy. Questions? Email: oosdata(at)usc.edu

---

## Before you start — OpenAI API setup

This app requires an OpenAI API key and a small amount of prepaid credits. **Read [OPENAI_API_SETUP.md](OPENAI_API_SETUP.md) first** — it walks through creating an account, generating a key, restricting its permissions, and turning off auto-recharge to protect against unexpected charges.

For pricing, go to https://developers.openai.com/api/docs/pricing, find **Specialized models**, and click **View more** to find `text-embedding-3-large` (hidden by default).

We find that a credit of just 5$ will last practically forever, as most runs are the cost of pennies (as of 5/13/26). 
---

## First-time setup (with separate instructions for Mac and Windows/Linux)

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

Getting embeddings from OpenAI is the only step that costs money. A checkpoint saves the result so you can reuse it on the same datasets for future runs without paying again (in case you want to adjust the model settings and rerun the same data).

- **Save checkpoint**: after embedding completes, the app writes a `.naics_embed` file.
- **Load checkpoint**: skips the embedding step entirely and reads a previously saved file. No API call or key required. Note!: Do not use a previously saved embedding for a new dataset. Each embedding file is unique to each dataset, and so you need to keep them organized based on each dataset name. 

**Notes on when to use saved embedding**: if you want to experiment with different training settings on the same dataset, save the checkpoint on the first run and load it each time you retrain. For a brand-new dataset you always need to embed, but you can save that result too. For prediction, the same logic applies — save if you might re-run the model on the same data.

---

## Billing

- **Start with a small test file** (10–20 rows) before running on your full dataset.
- **Keep auto-recharge OFF** and load only the minimum credits you need. See [OPENAI_API_SETUP.md](OPENAI_API_SETUP.md) for how to configure this.
- Monitor live usage at https://platform.openai.com/settings/organization/usage
