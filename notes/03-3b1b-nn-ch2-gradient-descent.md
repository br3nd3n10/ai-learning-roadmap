# 3Blue1Brown — Neural Networks ch. 2: Gradient descent, how neural networks learn

Video: https://www.3blue1brown.com/lessons/gradient-descent

## The big picture
- Chapter 1 built the machine (13,002 knobs); this chapter is about **how the knobs get tuned automatically**. "Learning" = minimizing a cost function.
- Start with totally **random weights and biases** — the network performs horribly — then improve it step by step using training data (MNIST: 60k labeled digit images).

## The cost function
- For one training example: compare the output layer to what it *should* be (label "3" → output neuron 3 should be 1, all others 0). **Cost of that example = sum of squared differences** between actual and desired activations. Small when confident and correct, large when wrong.
- The **total cost = average cost over all training examples**. Key mental shift: the cost is a **function of the 13,002 weights and biases** (the images are fixed parameters). Bad network → high cost; learning = finding the input (weight settings) that makes this function small.
- Layers of functions to keep straight: the *network* maps image → 10 scores; the *cost* maps 13,002 weights → one badness number.

## Gradient descent
- You can't solve "minimum of a 13,002-variable function" analytically. Instead: start anywhere, figure out which direction is **downhill**, take a small step, repeat — a ball rolling into a valley.
- From multivariable calculus: the **gradient** ∇C points in the direction of steepest *ascent*, so **step in the direction of −∇C** (steepest descent). Step size proportional to the slope helps avoid overshooting the minimum.
- You'll typically land in a **local minimum** — no guarantee it's the global one, but local minima tend to be good enough.
- This is **why activations are smooth/continuous** rather than binary like biological neurons firing: gradient descent needs a smoothly varying cost function to nudge downhill.

## What the gradient means
- −∇C is a 13,002-component vector: one nudge per weight/bias. **The magnitude of each component tells you how much that weight matters** — which knobs give the most cost-improvement per unit of nudge. Sign tells you which way to push it.
- **Backpropagation** (next two chapters) is the algorithm that computes this gradient efficiently.

## The humbling reality check
- The trained network works well: **~96% accuracy** on unseen images (~98% with tweaks) — genuinely impressive for something this simple.
- But peek inside: the hidden-layer neurons do **not** learn the clean edges and loops we hoped for in ch. 1 — the weight patterns look like noise. It works, but not for interpretable reasons.
- And it's overconfident: feed it **random noise** and it still confidently declares a digit. It knows how to classify digits, not how to *draw* or *recognize* them the way we do.
- Also: this plain multilayer perceptron is old technology (1980s-era) — the starting point for understanding modern nets, not the state of the art.

## Three lines to remember
1. Cost = average squared error over all training data, viewed as a function of the 13,002 weights/biases.
2. Learning = gradient descent: repeatedly step in the direction of −∇C, downhill on the cost surface.
3. It reaches ~96% accuracy, but the hidden layers don't learn human-interpretable features — working ≠ understanding.
