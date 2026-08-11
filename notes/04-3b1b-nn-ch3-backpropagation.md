# 3Blue1Brown — Neural Networks ch. 3: What is backpropagation really doing?

Video: https://www.3blue1brown.com/lessons/backpropagation

## The big picture
- Backpropagation is **the algorithm that computes the gradient** from ch. 2 — it answers "how should each of the 13,002 weights and biases change, and by how much, to decrease the cost most efficiently?"
- This chapter is the *intuition* without formulas; ch. 4 does the calculus.

## The core intuition: one training example's "wishes"
- Take a single example, an image of a **2**. The output layer is a mess. What we *want*: the "2" neuron's activation nudged **up**, all nine others nudged **down** — each nudge sized **in proportion to how far off** that neuron is (a very-wrong confident answer deserves a bigger correction than an almost-right one).
- There are exactly **three ways to increase a neuron's activation**:
  1. **Increase its bias**
  2. **Increase its weights** — most effective for weights connected to *bright* (high-activation) previous-layer neurons, since a(L−1) multiplies the weight. Echoes the Hebbian slogan: *"neurons that fire together wire together."*
  3. **Change the previous layer's activations** — brighten neurons with positive weights, dim ones with negative weights, in proportion to the weight sizes.

## The "back propagation" part
- You can't change activations directly — only weights and biases. But option 3 tells you what the **previous layer wishes it were**. Add up the wishes from **all ten output neurons** (the "2" neuron wants some hidden neurons brighter; the other nine want their own changes) to get one list of desired nudges for the second hidden layer.
- Then **recurse**: apply the same three-option logic to that layer, and propagate the desires backwards through the network. That backward flow of "wishes" is where the name comes from.

## From one example to the gradient
- One image of a 2 shouldn't dominate — the network would just classify everything as 2. So compute each example's desired nudges and **average them over all training data**: that average is (proportional to) the negative gradient from ch. 2.

## Stochastic gradient descent (SGD)
- Averaging over all 60k examples for every single step is painfully slow. Trick: **shuffle the data and split it into mini-batches** (e.g. 100 examples), compute the gradient on one mini-batch per step.
- Each step is only an *approximation* of the true downhill direction, but you take steps far faster. Analogy: a **drunk man stumbling quickly downhill** beats a careful man calculating the exact direction of steepest descent before each tiny step. This is the algorithm actually used in practice.

## One practical note
- All of this depends on having lots of **labeled training data** — the reason MNIST (and labeled datasets generally) mattered so much to the field.

## Three lines to remember
1. Each training example says how it wishes every output changed; those wishes translate into nudges for weights, biases, and (recursively) earlier layers — that backward recursion is backprop.
2. A weight's nudge scales with the activation feeding into it ("neurons that fire together wire together") and with how wrong the output is.
3. Averaged over the data these wishes = the negative gradient; mini-batches (SGD) make it fast at the price of noisy steps.
