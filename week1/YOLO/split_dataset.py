import os
import shutil
import random

DATASET_PATH = "/export/home/c5mcv02/KITTI-MOTS"
TRAINING_PATH = os.path.join(DATASET_PATH, "training")
TRAIN_OUTPUT = os.path.join(DATASET_PATH, "train")
VAL_OUTPUT = os.path.join(DATASET_PATH, "val")

# Create output folders
for split in [TRAIN_OUTPUT, VAL_OUTPUT]:
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "labels"), exist_ok=True)

sequences = sorted(os.listdir(os.path.join(TRAINING_PATH, "images")))

# 80% train, 20% val
random.seed(42)
random.shuffle(sequences)

split_index = int(0.8 * len(sequences))
train_seqs = sequences[:split_index]
val_seqs = sequences[split_index:]

def move_sequence(seq, split_path):
    seq_images_path = os.path.join(TRAINING_PATH, "images", seq)
    seq_labels_path = os.path.join(TRAINING_PATH, "labels", seq)

    # create subfolders
    os.makedirs(os.path.join(split_path, "images", seq), exist_ok=True)
    os.makedirs(os.path.join(split_path, "labels", seq), exist_ok=True)

    for img_file in sorted(os.listdir(seq_images_path)):
        shutil.move(os.path.join(seq_images_path, img_file), os.path.join(split_path, "images", seq, img_file))

    for lbl_file in sorted(os.listdir(seq_labels_path)):
        shutil.move(os.path.join(seq_labels_path, lbl_file), os.path.join(split_path, "labels", seq, lbl_file))

for seq in train_seqs:
    move_sequence(seq, TRAIN_OUTPUT)

for seq in val_seqs:
    move_sequence(seq, VAL_OUTPUT)

print(f"Completed: {len(train_seqs)} train sequences, {len(val_seqs)} val sequences")
