# Mini GPT

A small character-level GPT I built in PyTorch and trained on Tiny Shakespeare. There's no `nn.Transformer` or other ready-made attention in here: the attention heads, the blocks and the training loop are all written out by hand. It reads the last 256 characters and predicts the next one.

On the validation set it gets down to about 1.56 nats per character. A bigram model, which just counts which character follows which, gets 2.48, so that was the bar to beat.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivekpandey0706/Mini_GPT/blob/main/Mini_GPT_from_scratch.ipynb)

## Results

I held out the last 10% of the text (about 111k characters) for validation and never trained on it.

| Model | Params | Val loss | Bits/char | Perplexity |
|---|---|---|---|---|
| Bigram (counting) | 4,225 | 2.484 | 3.58 | 12.0 |
| Bigram (learned) | 4,225 | 2.476 | 3.57 | 11.9 |
| Mini GPT | 10.8M | 1.563 | 2.26 | 4.8 |

The loss is the average negative log-likelihood per character, so lower is better. Perplexity is e^loss, which is roughly how many characters the model is still torn between at each step. It goes from about 12 for the bigram to about 4.8.

![training curve](gpt_training_curve.png)

## Sample

Here's a stretch of what it writes when I start it from a single newline:

```text
KING RICHARD III:
To meet the king! God, and what stay hath now!
Your wife to give the keys, whose both you may mean:
Sleep the young Norfolk after'd like a sea,
And not like a miserable swue
That pass'd me in the public of the king.

Second Murderer:
An eagle, at villain.

First Murderer:
Though I thought, my liege.

Second Murderer:
How now, my lord, some other return me.

MAMILLIUS:
Now, by my life, I say this?

First Murderer:
If it be never spoke a fardel thing,
at your honour's disturb into the world.

Second Murderer:
What?

MENENIUS:
We must stab he,
Which therefore we start for this centurion,
No sequence. Even yet you to such a present name
Be cursed with unstrucking them! We'll not think up:
```

It picks up the format surprisingly quickly: real character names like King Richard III and Menenius, speaker names in capitals followed by a colon, verse-like line breaks, and mostly real words ("fardel" is even genuine Shakespeare). None of it actually means anything, and it makes up a word now and then ("swue", "unstrucking"). That makes sense, since it only ever sees 256 characters at a time and has no idea what a scene is about.

## Ablation

Before training the big model, I trained a few tiny ones to see where the improvement actually comes from. Each was 32-dim with an 8-character context, trained for 5,000 steps, and I added one piece at a time:

| What I added | Val loss |
|---|---|
| nothing (bigram) | 2.476 |
| a single attention head | 2.587 |
| 4 heads instead of 1 | 2.454 |
| a feed-forward layer | 2.609 |
| 3 blocks with residual connections | 2.258 |
| LayerNorm | 2.056 |
| scaling up (384-dim, 6 layers, 256 context, dropout) | 1.563 |

This didn't go the way I expected, and I think it's the most interesting result here.

- **A single attention head was worse than the bigram.** Without a residual connection, the current character only reaches the output if attention learns to look at itself. At the start, attention spreads its weight evenly over all the earlier positions, which blurs the one character that matters most.
- **Four heads helped**, but **adding a feed-forward layer made it worse again.** A deeper stack with no residual path is just harder to train.
- **Residual connections fixed it.** Once they went in, the 3-block model dropped to 2.26, and LayerNorm took it to 2.06. I added the extra blocks and the residuals in the same step, so I can't fully separate them. Still, the feed-forward result suggests the residuals are doing most of the work.
- **Scaling up gave the last big jump**, to 1.56.

## How it works

This is a rough walkthrough; the notebook has the details.

- **Tokens:** every distinct character is a token, 65 in total. I split by position (first 90% train, last 10% validation), so a passage can't end up half in each.
- **Batches:** random 256-character chunks, where the targets are the same chunk shifted by one. So each chunk is really 256 training examples at once.
- **Attention:** each position makes a query, a key and a value. The query-key scores, scaled by 1/sqrt(head size), go through a softmax that decides how much to read from each earlier position. A causal mask blocks the future. Without it, the model could just read the next character and cheat.
- **Block:** LayerNorm -> multi-head attention (6 heads of 64) -> add the input back, then LayerNorm -> feed-forward (384 -> 1536 -> 384) -> add the input back again. Dropout 0.2.
- **Model:** token embedding + position embedding, 6 blocks, a final LayerNorm, and a linear layer out to 65 logits.
- **Training:** AdamW, lr 3e-4, batches of 64 x 256, 5,000 steps, mixed precision on a free Colab T4.

The parameter count comes to 10,788,929:

| Part | Parameters |
|---|---|
| Embeddings | 123,264 |
| Each block | 1,773,312 |
| All 6 blocks | 10,639,872 |
| Final LayerNorm + output layer | 25,793 |

About two thirds of every block is the feed-forward layer, not the attention, which surprised me.

## Sanity checks

A few things I checked before spending 20+ minutes of GPU time:

- The untrained loss is about ln(65) = 4.17, so the model starts out guessing roughly uniformly.
- The parameter count matches the hand calculation above.
- Changing character 100 in the input doesn't change predictions 0-99 at all, so the mask works.
- The counted and learned bigrams agree (2.484 vs 2.476).

## Things I learned

- Attention is basically a weighted average where the weights depend on the content. The mask is what turns it into a language model.
- Residual connections matter more than I expected. Without them, adding a feed-forward layer made the small model worse (2.45 -> 2.61); with them, stacking 3 blocks brought it down to 2.26.
- The 1/sqrt(d) scaling isn't cosmetic. Without it the softmax starts out almost one-hot and learning stalls.
- 10.8M parameters on 1M characters overfits a bit, since train loss ends up below validation. Dropout helps; more data would help more.
- The loss and the sampling loop are exactly the same as in my [n-gram name generator](https://github.com/vivekpandey0706/bigram-trigram-name-generator). The only thing that changed is the model that computes the probabilities.

## What I'd do next

- Train longer, with a learning rate that decays over training. nanoGPT reaches about 1.47 on this setup that way.
- Swap characters for a BPE tokenizer, so 256 tokens cover a lot more text.
- Batch all the heads into one matmul (or use `F.scaled_dot_product_attention`), and add a KV cache so generation is faster.
- Tie the input embedding and output weights, then try a bigger dataset.

## Running it

Open the notebook in Colab, set the runtime to a T4 GPU, and hit Run all. It downloads the data itself, and the whole thing takes 30-40 minutes.

## Credits

I built this while following Andrej Karpathy's [Let's build GPT](https://karpathy.ai/zero-to-hero.html) video. The architecture and hyperparameters follow his [nanoGPT](https://github.com/karpathy/nanoGPT) Shakespeare setup. The original paper is Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and the data is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) from char-rnn.
