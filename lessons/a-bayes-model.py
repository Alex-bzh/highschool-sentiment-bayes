#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

##########
# Step 1 #
##########

# Define column names
cols = ['frequencies', 'lemma', 'tag', 'posScore', 'negScore']

# Load the file with tab separator
df_pos = pd.read_csv('data/positive.txt', sep='\t', header=None, names=cols)
df_neg = pd.read_csv('data/negative.txt', sep='\t', header=None, names=cols)

# Calculate the sum of all values in the 'frequencies' column
pos_sum = df_pos['frequencies'].sum()
neg_sum = df_neg['frequencies'].sum()

df = pd.concat([df_pos, df_neg], ignore_index=True)
df = df.groupby(['lemma', 'tag'], as_index=False).agg({
    'frequencies': 'sum',
})

V = df['frequencies'].sum()

# Define the probability function with a Laplacian smoothing
def P(frequencies, F, V):
    return (frequencies + 1) / (F + V)

ppos = df_pos['frequencies'].apply(lambda f: P(f, pos_sum, V))
pneg = df_neg['frequencies'].apply(lambda f: P(f, neg_sum, V))

df_pos['probabilities'] = ppos
df_neg['probabilities'] = pneg

##########
# Step 2 #
##########

df_vocab = pd.read_csv('data/vocabulary.txt', sep='\t')

# Filter df_vocab on label == 'positive'
df_vocab_pos = df_vocab[df_vocab['label'] == 'positive']
df_vocab_neg = df_vocab[df_vocab['label'] == 'negative']

# Merge on 'lemma' & 'tag' columns
df_pos = df_pos.merge(
  df_vocab_pos[['lemma', 'tag', 'weight']],
  on=['lemma', 'tag'],
  how='left'
)
df_neg = df_neg.merge(
  df_vocab_neg[['lemma', 'tag', 'weight']],
  on=['lemma', 'tag'],
  how='left'
)

df_pos['p_weighted'] = df_pos['probabilities'] * df_pos['weight'].fillna(1)
df_neg['p_weighted'] = df_neg['probabilities'] * df_neg['weight'].fillna(1)

##########
# Step 3 #
##########

# naive approach
prior_pos = 0.5
prior_neg = 0.5

# frequentist approach
prior_pos = pos_sum / (pos_sum + neg_sum)
prior_neg = neg_sum / (pos_sum + neg_sum)

# consistent approach
pos_weighted = (df_pos['frequencies'] * df_pos['weight']).sum()
neg_weighted = (df_neg['frequencies'] * df_neg['weight']).sum()

prior_pos = pos_weighted / (pos_weighted + neg_weighted)
prior_neg = neg_weighted / (pos_weighted + neg_weighted)

##########
# Step 4 #
##########

df = pd.read_csv("test/message1.txt", sep="\t")

# Keep only rows with non-zero conditional probability
df_pos = df[df["p_pos"] > 0.0].copy()
df_neg = df[df["p_neg"] > 0.0].copy()

df_pos["log_p_pos"] = np.log(df_pos["p_pos"])
df_neg["log_p_neg"] = np.log(df_neg["p_neg"])
df_pos["w_log_p_pos"] = df_pos["frequencies"] * df_pos["log_p_pos"]
df_neg["w_log_p_neg"] = df_neg["frequencies"] * df_neg["log_p_neg"]

log_likelihood_pos = df_pos["w_log_p_pos"].sum()
log_likelihood_neg = df_neg["w_log_p_neg"].sum()

score_pos = np.log(prior_pos) + log_likelihood_pos
score_neg = np.log(prior_neg) + log_likelihood_neg

# log-MAP scores
print(f"POS : {score_pos}, NEG : {score_neg}")

# probs
scores = np.array([score_pos, score_neg])
probs = np.exp(scores - np.max(scores))
probs /= probs.sum()

for label, prob in zip(["pos", "neg"], probs):
    print(f"{label}: {prob:.4f}")
