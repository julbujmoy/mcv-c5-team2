# Week 3: Image Captioning

This week, we started with the project of image captioning. We had a baseline that had ResNet 18 as a encoder and a GRU as a decoder. The aim of this project was to change the encoder and the decoder trying to achieve better results in generating text that descrbed the picture.

Link to the slides: https://docs.google.com/presentation/d/1k3j3JOhO4CImBR6ymKt4p-C3FjSkIGPyGcAsiFlpjaM/edit?usp=sharing

The pdf of the paper is in the week3 folder.

## Dataset: Food Images

We had food images and the caption is the title of the dish. The model needed to predict this caption.

## Objective and steps

1. Change the encoder: In our case, we changed it to ResNet 50.
2. Change the decoder: Chaged it to LSTM.
3. Experiment with different text representations: char, word or wordpiece.
4. Further improvements: hyperparameter tuning, attention, beam search, teacher forcing and improving post processing in inference.

### Step 1: Changing the encoder

The script that tested the ResNet 50 instead of the ResNet 18, you can execute it by:

```
python ResNet50-GRU.py
```

### Step 2: Changing the decoder

Execute the following:

```
python ResNet50-LSTM.py
```
We also tested it by just following the baseline and changing the decoder:

```
python ResNet18-LSTM.py
```

### Step 4: Applying attention, teacher forcing and beam search (beam width 3)

Execute the following, it has both ResNet 50 and LSTM with Adaptive attention:

```
python res50lstm1_attention.py
```

## Results

We began by testing a baseline model using a ResNet18 encoder and GRU decoder across three text representations: character-level, word-level, and WordPiece.Among these, WordPiece consistently outperformed the others, offering better token handling and semantinc structure. Building on this, we replaced the encoder with ResNet50, which provided richer visual features, and later swapped the decoder from GRU to LSTM, finding that a single-layer LSTM gave the best results overall. Through successive improvements; including attention, teacher orcing, refined loss handling, decoding with beam search, and cleaner post processing, we observed steady gains in fluency, coherence, and lexical diversity. The final pipeline (ResNet50 + 1-layer LSTM + attention + WordPiece ++ Beam Search) achieved the best overall performance, balancing lower training loss with significantly improved metrics such as BLEU-2, ROUGE, and METEOR, and producing more structured and semantically meaningful captions.


