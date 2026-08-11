# 3Blue1Brown — Neural Networks ch. 5: But what is a GPT? Visual intro to transformers

Video: https://www.3blue1brown.com/lessons/gpt

## The big picture
- **GPT = Generative Pretrained Transformer.** Generative: it produces text. Pretrained: learned from massive data, then adaptable. **Transformer: the specific neural network architecture** — the invention (Google, 2017, "Attention Is All You Need") behind the current AI boom.
- The whole model does one thing: take in a sequence of text and produce a **probability distribution over what token comes next**. Chatbots are just this, run in a loop: predict, **sample** a token from the distribution, append it, repeat. (Ties back to Karpathy's "dreaming" — and explains why output is different every run.)

## The pipeline (the map to memorize — chapters 6–7 zoom into the middle)
1. **Tokenization** — text is chopped into **tokens** (words/chunks of words; ~50k vocabulary for GPT-3).
2. **Embedding** — each token becomes a **vector** via the embedding matrix. GPT-3: 12,288 dimensions.
3. **Attention blocks** — vectors talk to each other and update each other based on context. (Chapter 6.)
4. **MLP / feed-forward blocks** — each vector is processed individually, no cross-talk. (Chapter 7.)
5. Repeat 3–4 many times, alternating.
6. **Unembedding** — the **last vector** of the sequence is mapped (by the unembedding matrix) to ~50k **logits**, one per possible next token.
7. **Softmax** — turns logits into a probability distribution.

## Embeddings: meaning as directions (the key concept of this chapter)
- A word's embedding is a point in very-high-dimensional space, and **directions in that space carry meaning**:
  - The classic: **E(king) − E(man) + E(woman) ≈ E(queen)** — one direction encodes gender. Similar: E(Germany) − E(Japan) + E(sushi) ≈ E(bratwurst).
  - **Dot product measures similarity/alignment** — e.g. plurality: (E(cats) − E(cat)) dotted against other words scores how "plural" they are.
- Embeddings start as one-vector-per-token from training, but the network's job is to **enrich them with context** as they flow through the layers — "king" should become "a king who lives in Scotland, in a Shakespeare play…" Meaning gets baked into the vector, not the word.
- The vector also encodes the token's **position** in the sequence.

## Context window and the final prediction
- The **context size** (GPT-3: 2048 tokens) is how many vectors flow through at once — the hard limit on how much the model can "keep in mind" (why early long chats seemed to forget the beginning).
- Only the **final vector** in the sequence is used to predict the next token — by the last layer, it has to have absorbed the meaning of the whole passage. (During training, predicting at *every* position gives more feedback per pass.)

## Softmax with temperature (worth writing down)
- Softmax: exponentiate each logit, divide by the sum — turns any list of numbers into a valid probability distribution where big inputs dominate.
- **Temperature T** divides the logits first: **T = 0** → always pick the max (deterministic, safe, boring); **higher T** → flatter distribution, more diverse/riskier output. This is the actual meaning of the "temperature" API parameter I've used in apps.

## Scale of the thing
- GPT-3: **175 billion parameters**, organized into **27,938 matrices**. All of them tuned by the same backprop/gradient-descent story from chapters 2–4 — nothing new in *how* it learns, only in *what* the architecture is.
- Deep learning's common shape: input as an array of numbers, layers of tunable weight matrices multiplying activations, trained end-to-end by backprop.

## Three lines to remember
1. A GPT maps token sequence → probability distribution over the next token; chat = predict, sample, append, repeat.
2. Tokens become high-dimensional vectors where directions encode meaning (king − man + woman ≈ queen), and the layers' job is to soak context into those vectors.
3. Pipeline: tokenize → embed → [attention ↔ MLP] × many → unembed last vector → softmax (with temperature) → next-token odds.
