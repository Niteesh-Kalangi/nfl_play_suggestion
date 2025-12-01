# Quick Commands: Push to GitHub & Run in Colab

## Step 1: Prepare Data (if needed)

Since `data/*.csv` is in `.gitignore`, you'll need to upload it separately to Colab:

```bash
cd /Users/sashankt19/Projects/nfl_play_suggestion
zip -r data.zip data/
```

Then upload `data.zip` to Google Drive (you'll use it in Colab).

## Step 2: Push Code to GitHub

```bash
# Navigate to project
cd /Users/sashankt19/Projects/nfl_play_suggestion

# Check what will be committed
git status

# Add all changes
git add .

# Commit
git commit -m "Update autoregressive models to match diffusion structure

- Added new [T, P, F] data format (22 players, 3 features)
- Added context encoder for categorical/continuous conditioning
- Updated LSTM and Transformer models
- Added visualization utilities
- Added custom play generation API
- Created Colab training notebook
- Disabled early stopping (train all 50 epochs)"

# Push to GitHub
git push origin autoregression
```

## Step 3: Set Up Colab

1. **Open Google Colab**: https://colab.research.google.com
2. **Upload notebook**: File → Upload notebook → Select `colab_train.ipynb`
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)

## Step 4: Configure Colab Notebook

In **Cell 2**, update these lines:

```python
GITHUB_REPO = "https://github.com/YOUR_USERNAME/nfl_play_suggestion.git"  # Change this!
BRANCH = "autoregression"  # Change if different
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 5: Run in Colab

Run cells in order (Shift+Enter or click play button):

1. **Cell 1-2**: Setup & clone repository
2. **Cell 3-4**: Load data from Google Drive (if needed)
   - Update `data_zip_path` in Cell 4 to where you uploaded `data.zip`
3. **Cell 5-6**: Verify data files
4. **Cell 7-8**: Train models (30-60 minutes)
5. **Cell 9-10**: Load models
6. **Cell 11-12**: Generate and visualize plays
7. **Cell 13-14**: Animated visualization (optional)
8. **Cell 15-16**: Download results

## That's It! 🚀

For detailed instructions, see `COLAB_STEP_BY_STEP.md`
