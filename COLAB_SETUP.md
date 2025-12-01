# Running on Google Colab with GPU

This guide helps you run the autoregressive training on Google Colab with free GPU.

## Step 1: Setup Your Project in Colab

### Option A: Clone from GitHub (Recommended - No Upload Needed!)

1. **Push your project to GitHub:**
   ```bash
   cd /Users/sashankt19/Projects/nfl_play_suggestion
   git add .
   git commit -m "Ready for Colab training"
   git push origin autoregression  # or your branch name
   ```

2. **In Colab notebook, update the GitHub URL:**
   - Open `colab_train.ipynb` in Colab
   - Find `GITHUB_REPO = "https://github.com/yourusername/nfl_play_suggestion.git"`
   - Replace with your actual GitHub repository URL
   - Update `BRANCH = "autoregression"` if your branch is different

3. **That's it!** The notebook will clone it automatically.

### Option B: Upload to Google Drive (Alternative)

1. **Zip your project:**
   ```bash
   cd /Users/sashankt19/Projects
   zip -r nfl_play_suggestion.zip nfl_play_suggestion/
   ```

2. **Upload to Google Drive:**
   - Go to https://drive.google.com
   - Upload `nfl_play_suggestion.zip`
   - Note the location

3. **In Colab notebook, comment out GitHub section and uncomment Drive section**

## Step 2: Open Colab Notebook

1. Go to https://colab.research.google.com
2. **Upload the notebook:**
   - File → Upload notebook
   - Upload `colab_train.ipynb` from your project
   - OR copy-paste the cells from the notebook
3. **Enable GPU:** Runtime → Change runtime type → GPU (T4 or better)
4. **Update GitHub URL** in the first cell (if using GitHub method)
5. Run the cells:

### Cell 1: Setup (GitHub or Drive)
```python
# OPTION A: Clone from GitHub (Recommended)
GITHUB_REPO = "https://github.com/yourusername/nfl_play_suggestion.git"
BRANCH = "autoregression"

import os, subprocess
if not os.path.exists('nfl_play_suggestion'):
    subprocess.run(['git', 'clone', '-b', BRANCH, GITHUB_REPO], check=True)
os.chdir('nfl_play_suggestion')

# OPTION B: Use Google Drive (Alternative)
# from google.colab import drive
# drive.mount('/content/drive')
# os.chdir('/content/drive/MyDrive/nfl_play_suggestion')

%pip install -q torch pandas numpy matplotlib tqdm pyyaml ipython
```

### Cell 2: Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 3: Train Model
```python
!python train_autoregressive.py --model lstm --device cuda
```

### Cell 4: Download Results (after training)
```python
from google.colab import files

# Download model checkpoint
files.download('artifacts/autoregressive/lstm.pt')

# Download training history
files.download('artifacts/autoregressive/lstm_history.json')

# Download results
files.download('artifacts/autoregressive/results.json')
```

## Step 3: Quick Colab Notebook Template

Here's a complete notebook you can copy-paste:

```python
# ============================================================================
# CELL 1: Setup
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')

!pip install -q torch pandas numpy matplotlib tqdm pyyaml

import os
import sys
os.chdir('/content/drive/MyDrive/nfl_play_suggestion')  # UPDATE THIS PATH
sys.path.insert(0, os.getcwd())

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CELL 2: Train LSTM
# ============================================================================
!python train_autoregressive.py --model lstm --device cuda

# ============================================================================
# CELL 3: Train Transformer (optional)
# ============================================================================
!python train_autoregressive.py --model transformer --device cuda

# ============================================================================
# CELL 4: Generate and Visualize Play
# ============================================================================
!python generate_play.py \
    --model lstm \
    --checkpoint artifacts/autoregressive/lstm.pt \
    --device cuda \
    --down 1 \
    --yards_to_go 10 \
    --formation SHOTGUN \
    --personnel "1 RB, 1 TE, 3 WR" \
    --animate \
    --save play.mp4

# Display animation
from IPython.display import Video
Video('play.mp4')

# ============================================================================
# CELL 5: Download Results
# ============================================================================
from google.colab import files

files.download('artifacts/autoregressive/lstm.pt')
files.download('artifacts/autoregressive/lstm_history.json')
files.download('artifacts/autoregressive/results.json')
```

## Important Notes

1. **Update the path** in Cell 1 to match where you uploaded your project
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Session timeout**: Colab sessions timeout after ~12 hours of inactivity
4. **Data location**: Make sure your `data/` folder is in the project directory

## Faster Alternative: Direct Upload

If you don't want to use Drive, you can upload directly:

```python
# Upload project zip
from google.colab import files
uploaded = files.upload()  # Upload nfl_play_suggestion.zip

# Extract
!unzip -q nfl_play_suggestion.zip
os.chdir('nfl_play_suggestion')
```

## GPU vs CPU

**GPU is MUCH better for training!**

| Device | Training Time (50 epochs) | Speed |
|--------|---------------------------|-------|
| **Colab GPU (T4)** | ~30-60 minutes | ⚡⚡⚡ Fastest |
| **Colab GPU (A100)** | ~15-30 minutes | ⚡⚡⚡⚡ Fastest |
| **Local GPU (CUDA)** | ~30-60 minutes | ⚡⚡⚡ Fast |
| **Local MPS (Mac)** | ~2-4 hours | ⚡⚡ Medium |
| **CPU** | ~4-8 hours | ⚡ Slow |

**Always use GPU if available!** The speedup is 10-20x faster.

## Speed Comparison

- **Local MPS (Mac)**: ~2-4 hours for 50 epochs
- **Colab GPU (T4)**: ~30-60 minutes for 50 epochs  
- **Colab GPU (A100)**: ~15-30 minutes for 50 epochs

Colab GPU will be **much faster**!

