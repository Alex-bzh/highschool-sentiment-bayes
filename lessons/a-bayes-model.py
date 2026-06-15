#!/usr/bin/env python
# coding: utf-8

import argparse
import pandas as pd
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("pathfile", help="Path to the message file")
args = parser.parse_args()

##########
# Step 1 #
##########

# Load the file with tab separator
df = pd.read_csv("./data/vocabulary.txt", sep="\t")

# Calculate the sum of all values in the 'frequencies' column
pos_sum = df["posFreq"].sum()
neg_sum = df["negFreq"].sum()

# Vocabulary length
V = len(df)

# Define the probability function with a Laplacian smoothing
def P(frequencies, F, V):
    return (frequencies + 1) / (F + V)

# Apply 'P' function to the values in the 'posFreq' column …
ppos = df["posFreq"].apply(lambda f: P(f, pos_sum, V))
pneg = df["negFreq"].apply(lambda f: P(f, neg_sum, V))

# … and add them to the data frame
df["posProb"] = ppos
df["negProb"] = pneg

##########
# Step 2 #
##########

# Calculate prior with the frequentist approach
prior_pos = pos_sum / V
prior_neg = neg_sum / V

##########
# Step 3 #
##########

# Load message file provided via CLI
df_msg = pd.read_csv(args.pathfile, sep="\t")

# Add the probabilities learned from the corpus
df_msg = pd.merge(
    df_msg,
    df[["lemma", "tag", "posProb", "negProb"]],
    on=["lemma", "tag"],
    how="left"
)

# Drop useless information
df_msg = df_msg.dropna(subset=["negProb", "posProb"], how="all")

# Convert into log-probabilities
log_pos_prob = np.log(df_msg["posProb"])
log_neg_prob = np.log(df_msg["negProb"])

# Calculate the log-likelihood
log_likelihood_pos = df_msg["freq"] * log_pos_prob
log_likelihood_neg = df_msg["freq"] * log_neg_prob

# Add log-prior to the log-likelihood to obtain a score
score_pos = np.log(prior_pos) + log_likelihood_pos.sum()
score_neg = np.log(prior_neg) + log_likelihood_neg.sum()

# log-MAP scores
print(f"POS : {score_pos}, NEG : {score_neg}")

# Convert log-MAP score into probabilities
scores = np.array([score_pos, score_neg])
probs = np.exp(scores - np.max(scores))
probs /= probs.sum()

for label, prob in zip(["pos", "neg"], probs):
    print(f"{label}: {prob:.4f}")
