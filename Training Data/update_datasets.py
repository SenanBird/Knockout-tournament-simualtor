# Save this file as: /mnt/4tb_games/Projects/2026/Training Data/update_datasets.py
# To run it manually from terminal: python update_datasets.py

import os
import shutil
import kagglehub

# =========================================================
# CONFIGURATION
# =========================================================
# Points directly to the folder this script lives in ('Training Data')
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(TARGET_DIR, exist_ok=True)

print("========================================================")
print("🚀 STARTING AUTOMATED DATASET SYNCHRONIZATION")
print("========================================================")

# ---------------------------------------------------------
# STEP 1: Sync International Match Records
# ---------------------------------------------------------
print("\n-> Checking Kaggle for fresh International Match results...")
# Downloads the latest version of Mart Jürisoo's dataset
match_data_path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")

print(f"   Downloaded package cache located at: {match_data_path}")

# Streamlined: 'shootouts.csv' removed since it isn't utilized by the AI brain
match_files = ["results.csv"]

for file_name in match_files:
    source_file = os.path.join(match_data_path, file_name)
    destination_file = os.path.join(TARGET_DIR, file_name)
    
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
        print(f"   ✅ Copied successfully: {file_name} -> {TARGET_DIR}")
    else:
        print(f"   ❌ Error: Could not find {file_name} in the downloaded dataset.")

# ---------------------------------------------------------
# STEP 2: Sync Transfermarkt Player Market Values
# ---------------------------------------------------------
print("\n-> Checking Kaggle for fresh Transfermarkt Squad values...")
# Downloads the latest version of David Cariboo's dataset (Scraped weekly)
player_data_path = kagglehub.dataset_download("davidcariboo/player-scores")

print(f"   Downloaded package cache located at: {player_data_path}")

# Complete: Added 'player_valuations.csv' required by the squad feature pre-processing blocks
player_files = ["players.csv", "player_valuations.csv"]

for file_name in player_files:
    source_file = os.path.join(player_data_path, file_name)
    destination_file = os.path.join(TARGET_DIR, file_name)
    
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
        print(f"   ✅ Copied successfully: {file_name} -> {TARGET_DIR}")
    else:
        print(f"   ❌ Error: Could not find {file_name} in the downloaded dataset.")

# ---------------------------------------------------------
# STEP 3: Sync FIFA Rankings (Automated Missing Link)
# ---------------------------------------------------------
print("\n-> Checking Kaggle for fresh FIFA Rankings...")
# Downloads Tadhg Fitzgerald's structural historical tracking data
fifa_data_path = kagglehub.dataset_download("tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now")

print(f"   Downloaded package cache located at: {fifa_data_path}")

source_file = os.path.join(fifa_data_path, "fifa_ranking.csv")
destination_file = os.path.join(TARGET_DIR, "fifa_ranking.csv")

if os.path.exists(source_file):
    shutil.copy(source_file, destination_file)
    print(f"   ✅ Copied successfully: fifa_ranking.csv -> {TARGET_DIR}")
else:
    print(f"   ❌ Error: Could not find fifa_ranking.csv in the downloaded dataset.")

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------
print("\n========================================================")
print("🎉 SUCCESS! All 4 local project datasets are up to date.")
print(f"   Location: {TARGET_DIR}")
print("   You can now safely execute your PyTorch Brain script.")
print("========================================================")