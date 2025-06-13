#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

##########
# Step 1 #
##########

# Load the file with tab separator
df = pd.read_csv('../data/vocabulary.txt', sep='\t')

# Calculate the sum of all values in the 'frequencies' column
pos_sum = df['posFreq'].sum()
neg_sum = df['negFreq'].sum()

V = pos_sum + neg_sum

# Define the probability function with a Laplacian smoothing
def P(frequencies, F, V):
    return (frequencies + 1) / (F + V)

ppos = df['posFreq'].apply(lambda f: P(f, pos_sum, V))
pneg = df['negFreq'].apply(lambda f: P(f, neg_sum, V))

df['posProb'] = ppos
df['negProb'] = pneg

##########
# Step 2 #
##########

df['posProb_w'] = df['posProb'] * df['posWeight']
df['negProb_w'] = df['negProb'] * df['negWeight']

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
pos_weighted = (df['posFreq'] * df['posWeight']).sum()
neg_weighted = (df['negFreq'] * df['negWeight']).sum()

prior_pos = pos_weighted / (pos_weighted + neg_weighted)
prior_neg = neg_weighted / (pos_weighted + neg_weighted)

##########
# Step 4 #
##########

# read message
df_msg = pd.read_csv("../test/message1.txt", sep="\t")

df_msg = pd.merge(
    df_msg,
    df[['lemma', 'tag', 'posProb_w', 'negProb_w']],
    on=['lemma', 'tag'],
    how='left'
)

df_msg = df_msg.dropna(subset=['negProb_w', 'posProb_w'], how='all')

alpha = 1e-6

df_msg['posProb_w'] = df_msg['posProb_w'] + alpha
df_msg['negProb_w'] = df_msg['negProb_w'] + alpha

log_pos_prob = np.log(df_msg["posProb_w"])
log_neg_prob = np.log(df_msg["negProb_w"])

log_likelihood_pos = df_msg["freq"] * log_pos_prob
log_likelihood_neg = df_msg["freq"] * log_neg_prob

sum_log_likelihood_pos = log_likelihood_pos.sum()
sum_log_likelihood_neg = log_likelihood_neg.sum()

score_pos = np.log(prior_pos) + sum_log_likelihood_pos
score_neg = np.log(prior_neg) + sum_log_likelihood_neg

# log-MAP scores
print(f"POS : {score_pos}, NEG : {score_neg}")

# probs
scores = np.array([score_pos, score_neg])
probs = np.exp(scores - np.max(scores))
probs /= probs.sum()

for label, prob in zip(["pos", "neg"], probs):
    print(f"{label}: {prob:.4f}")
