# Step-by-Step Guide: From Local to Colab Training

Complete guide to push your code to GitHub and train in Colab.

## Part 1: Push Changes to GitHub

### Step 1: Check your current git status
```bash
cd /Users/sashankt19/Projects/nfl_play_suggestion
git status
```

### Step 2: Add all the new files
```bash
git add .
```

This adds:
- `src/autoregressive/` (new directory with all the updated code)
- `train_autoregressive.py` (updated training script)
- `generate_play.py` (new generation script)
- `config.yaml` (updated config)
- `colab_train.ipynb` (Colab notebook)
- All other new/modified files

### Step 3: Commit the changes
```bash
git commit -m "Update autoregressive models to match diffusion structure

- Added new [T, P, F] data format (22 players, 3 features)
- Added context encoder for categorical/continuous conditioning
- Updated LSTM and Transformer models
- Added visualization utilities
- Added custom play generation API
- Created Colab training notebook
- Disabled early stopping (train all 50 epochs)"
```

### Step 4: Push to GitHub
```bash
git push origin autoregression
```

**Note:** If this is your first push to this branch, you might need:
```bash
git push -u origin autoregression
```

---

## Part 2: Set Up Colab

### Step 5: Open Google Colab
1. Go to https://colab.research.google.com
2. Sign in with your Google account

### Step 6: Upload the Notebook
1. Click **File** → **Upload notebook**
2. Navigate to your project: `/Users/sashankt19/Projects/nfl_play_suggestion/`
3. Select `colab_train.ipynb`
4. Click **Upload**

**OR** create a new notebook and copy-paste cells from `colab_train.ipynb`

### Step 7: Enable GPU
1. In Colab, click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU** (T4 or better)
3. Click **Save**

---

## Part 3: Configure and Run in Colab

### Step 8: Update GitHub URL in Colab

In the first code cell (Cell 2), find this line:
```python
GITHUB_REPO = "https://github.com/yourusername/nfl_play_suggestion.git"  # Change this!
```

**Replace with your actual GitHub repository URL:**
- If your repo is: `https://github.com/sashankt19/nfl_play_suggestion`
- Then set: `GITHUB_REPO = "https://github.com/sashankt19/nfl_play_suggestion.git"`

**Also check the branch name:**
```python
BRANCH = "autoregression"  # Change if your branch is different
```

### Step 9: Run Cells Sequentially

Run each cell in order by clicking the play button or pressing `Shift+Enter`:

#### Cell 1: Setup (Run this first)
- Clones your repository from GitHub
- Installs dependencies
- Checks GPU availability
- **Expected output:** "✅ Repository cloned!" and GPU info

#### Cell 2: Verify Data
- Checks that data files exist
- **Expected output:** "✅ Found X files in data/" and "✅ All required files present"

#### Cell 3: Train Models
- Trains both LSTM and Transformer
- **This takes 30-60 minutes on GPU**
- **Expected output:** Training progress bars, epoch-by-epoch metrics

#### Cell 4: Load Models
- Loads trained models for generation
- **Expected output:** "✅ LSTM model loaded" and optionally "✅ Transformer model loaded"

#### Cell 5: Generate Play
- Generates a play with custom parameters
- Shows visualization
- **You can modify the parameters in this cell!**

#### Cell 6: Animated Visualization (Optional)
- Creates animated video of the play
- **Expected output:** Animation saved and displayed

#### Cell 7: Download Results
- Downloads trained models to your computer
- **Expected output:** Files download automatically

---

## Part 4: Data Files Issue

**⚠️ IMPORTANT:** Your `.gitignore` excludes `data/*.csv` files, so they won't be in GitHub.

**You have 2 options:**

### Option A: Upload Data to Google Drive (Recommended)
1. Zip your data folder: `zip -r data.zip data/`
2. Upload `data.zip` to Google Drive
3. In Colab, after cloning, add this cell:
```python
# Mount Drive and extract data
from google.colab import drive
drive.mount('/content/drive')

# Copy data.zip from Drive to Colab
!cp /content/drive/MyDrive/data.zip /content/nfl_play_suggestion/
!cd /content/nfl_play_suggestion && unzip -q data.zip
```

### Option B: Use Google Drive for Everything
Use the Google Drive method in Cell 2 instead of GitHub (uncomment Option B section)

---

## Part 5: Troubleshooting

### Issue: "Repository not found" or "Permission denied"
**Solution:** 
- Check your GitHub URL is correct
- Make sure the repository is public, OR
- Use a personal access token for private repos

### Issue: "data/ directory not found"
**Solution:**
- Check if `data/` is in `.gitignore`
- If yes, either commit it or use Google Drive method
- To commit: `git add data/` then `git commit -m "Add data files"`

### Issue: "No GPU available"
**Solution:**
- Runtime → Change runtime type → GPU
- Wait a few minutes if GPU is not immediately available

### Issue: Training is slow
**Solution:**
- Make sure GPU is enabled (check Cell 1 output)
- Reduce batch size in `config.yaml` if out of memory
- Reduce epochs for testing (change `epochs: 50` to `epochs: 10`)

---

## Quick Reference Commands

### Local (Terminal):
```bash
# Navigate to project
cd /Users/sashankt19/Projects/nfl_play_suggestion

# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push origin autoregression
```

### Colab (Notebook):
1. Update `GITHUB_REPO` in Cell 2
2. Run Cell 1 (Setup)
3. Run Cell 2 (Verify Data)
4. Run Cell 3 (Train) - **Wait 30-60 minutes**
5. Run Cell 4 (Load Models)
6. Run Cell 5 (Generate Play)
7. Run Cell 6 (Animate - Optional)
8. Run Cell 7 (Download)

---

## Expected Timeline

- **Setup & Data Check:** 2-5 minutes
- **Training (both models):** 30-60 minutes on GPU
- **Generation & Visualization:** 1-2 minutes per play
- **Total:** ~1-2 hours for complete training + generation

---

## Next Steps After Training

Once training completes:
1. Models are saved to `artifacts/autoregressive/`
2. You can generate unlimited plays by modifying Cell 5
3. Download models to use locally
4. Compare with diffusion model results

Good luck! 🚀

