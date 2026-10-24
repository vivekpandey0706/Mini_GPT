# Mini GPT

A small character-level GPT I built in PyTorch and trained on Tiny Shakespeare. There's no `nn.Transformer` or other ready-made attention in here: the attention heads, the blocks and the training loop are all written out by hand. It reads the last 256 characters and predicts the next one.

On the validation set it gets down to about 1.48 nats per character. A bigram model, which just counts which character follows which, gets 2.48, so that was the bar to beat. <!-- TODO: your final val loss from Cell 14 -->

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivekpandey0706/mini-gpt/blob/main/Mini_GPT.ipynb)

## Results

I held out the last 10% of the text (about 111k characters) for validation and never trained on it.

| Model | Params | Val loss | Bits/char | Perplexity |
|---|---|---|---|---|
| Bigram (counting) | 4,225 | 2.484 | 3.58 | 12.0 |
| Bigram (learned) | 4,225 | 2.49 | 3.59 | 12.1 |
| Mini GPT | 10.8M | 1.48 | 2.14 | 4.4 |
<!-- TODO: replace the last two rows with your Cell 14 table -->

The loss is the average negative log-likelihood per character, so lower is better. Perplexity is e^loss, which is roughly how many characters the model is still torn between at each step. It goes from about 12 for the bigram to about 4.4.

![training curve](gpt_training_curve.png)

## Sample

Here's what it writes when I start it from a single newline:

```text
(paste ~15 lines from generated_shakespeare.txt here)
```

It picks up the format surprisingly quickly: names in capitals followed by a colon, line breaks that look like verse, mostly real words. None of it actually means anything. That makes sense, since it only ever sees 256 characters at a time and has no idea what a scene is about.

## Ablation

Before training the big model, I trained a few tiny ones to see where the improvement actually comes from. Each was 32-dim with an 8-character context, trained for 5,000 steps, and I added one piece at a time:

| What I added | Val loss |
|---|---|
| nothing (bigram) | 2.49 |
| a single attention head | 2.40 |
| 4 heads instead of 1 | 2.30 |
| a feed-forward layer | 2.25 |
| 3 blocks with residual connections | 2.10 |
| LayerNorm | 2.05 |
| scaling up (384-dim, 6 layers, 256 context, dropout) | 1.48 |
<!-- TODO: replace with your Cell 10 output -->

Attention on its own only helps a little. Stacking blocks didn't really pay off until the residual connections were in, and the biggest jump by far came from simply making everything bigger.

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
- The counted and learned bigrams agree (2.484 vs ~2.49).

## Things I learned

- Attention is basically a weighted average where the weights depend on the content. The mask is what turns it into a language model.
- The 1/sqrt(d) scaling isn't cosmetic. Without it the softmax starts out almost one-hot and learning stalls.
- Residuals and LayerNorm are what make depth work; the ablation shows it pretty clearly.
- 10.8M parameters on 1M characters overfits a bit, since train loss ends up below validation. Dropout helps; more data would help more.
- The loss and the sampling loop are exactly the same as in my [n-gram name generator](https://github.com/vivekpandey0706/bigram-trigram-name-generator). The only thing that changed is the model that computes the probabilities.

## What I'd do next

- Swap characters for a BPE tokenizer, so 256 tokens cover a lot more text.
- Batch all the heads into one matmul (or use `F.scaled_dot_product_attention`), and add a KV cache so generation is faster.
- Tie the input embedding and output weights, then try a bigger dataset.

## Running it

Open the notebook in Colab, set the runtime to a T4 GPU, and hit Run all. It downloads the data itself, and the whole thing takes 30-40 minutes.

## Credits

I built this while following Andrej Karpathy's [Let's build GPT](https://karpathy.ai/zero-to-hero.html) video. The architecture and hyperparameters follow his [nanoGPT](https://github.com/karpathy/nanoGPT) Shakespeare setup. The original paper is Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762), and the data is [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) from char-rnn.
