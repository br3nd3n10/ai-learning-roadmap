# 3Blue1Brown — Neural Networks ch. 4: Backpropagation calculus

Video: https://www.3blue1brown.com/lessons/backpropagation-calculus

## The big picture
- Same idea as ch. 3, now as actual formulas. The whole chapter is **the chain rule applied to a network** — and the trick that makes it digestible is starting with the simplest possible network: **one neuron per layer**.

## Setup and notation (worth copying down exactly)
For the last layer L, with one neuron per layer:

- **z⁽ᴸ⁾ = w⁽ᴸ⁾ a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾**  (weighted input, before squishing)
- **a⁽ᴸ⁾ = σ(z⁽ᴸ⁾)**  (activation)
- **C₀ = (a⁽ᴸ⁾ − y)²**  (cost of one training example; y = desired output)

Keep the dependency chain in your head: **w → z → a → C**. Each arrow is one derivative; the chain rule multiplies them.

## The three key derivatives
Sensitivity of the cost to the last-layer **weight**:

  ∂C₀/∂w⁽ᴸ⁾ = ∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂C₀/∂a⁽ᴸ⁾ = **a⁽ᴸ⁻¹⁾ · σ′(z⁽ᴸ⁾) · 2(a⁽ᴸ⁾ − y)**

Read each factor as intuition, matching ch. 3:
- **a⁽ᴸ⁻¹⁾** — the weight matters in proportion to how bright the neuron feeding it is (fire together, wire together)
- **σ′(z⁽ᴸ⁾)** — matters more when the neuron is in the steep, sensitive part of the squishing function
- **2(a⁽ᴸ⁾ − y)** — matters in proportion to how wrong the output is

For the **bias**: identical except ∂z/∂b = 1, so **∂C₀/∂b⁽ᴸ⁾ = σ′(z⁽ᴸ⁾) · 2(a⁽ᴸ⁾ − y)**.

For the **previous activation**: ∂z/∂a⁽ᴸ⁻¹⁾ = w⁽ᴸ⁾, so **∂C₀/∂a⁽ᴸ⁻¹⁾ = w⁽ᴸ⁾ · σ′(z⁽ᴸ⁾) · 2(a⁽ᴸ⁾ − y)**.

## Why the third one is the special one
- You can't change a⁽ᴸ⁻¹⁾ directly, but ∂C₀/∂a⁽ᴸ⁻¹⁾ is the hook for **recursion**: once you know how sensitive the cost is to a layer's activation, you repeat the exact same chain-rule step to get the sensitivities of the weights and biases *behind* it, and keep walking backwards. That backward walk **is** backpropagation.

## Scaling up to real layers
- With many neurons per layer, almost nothing changes — just more indices: wⱼₖ⁽ᴸ⁾ is the weight from neuron k in layer L−1 to neuron j in layer L; the cost sums over output neurons.
- The one real addition: a hidden neuron influences the cost **through multiple paths** (it feeds every neuron in the next layer), so ∂C/∂a⁽ᴸ⁻¹⁾ becomes a **sum over those paths**. Everything else is the same formulas with subscripts.

## Wrapping up
- Average these per-example derivatives over training data (in practice, over a mini-batch) and you have every component of ∇C — the gradient descent step from ch. 2 is now fully specified.
- Sit with the chain rule until each factor feels like the ch. 3 intuition restated; that's the whole game. This is the mathematical core of how every neural network learns.

## Three lines to remember
1. Dependency chain w → z → a → C; the chain rule multiplies one derivative per link.
2. ∂C/∂w = a⁽ᴸ⁻¹⁾ · σ′(z) · 2(a − y) — input brightness × squish sensitivity × how wrong you are.
3. ∂C/∂a⁽ᴸ⁻¹⁾ (summed over paths) is what lets the same step recurse backwards through every layer.
