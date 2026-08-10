# 3Blue1Brown — Neural Networks ch. 1: But what is a neural network?

Video: https://www.3blue1brown.com/lessons/neural-networks

## The big picture
- A neural network is just a **function**: it takes numbers in (784 pixel values) and gives numbers out (10 digit scores). Everything else is detail about how that function is built.
- Running example: **MNIST digit recognition** — a 28×28 grayscale image → which digit 0–9 is it?

## Neurons and layers
- A **neuron is just a thing that holds a number** between 0 and 1, called its **activation**.
- Structure of the example network:
  - **Input layer:** 784 neurons (one per pixel, activation = brightness 0–1)
  - **2 hidden layers:** 16 neurons each (an arbitrary choice, picked for illustration)
  - **Output layer:** 10 neurons, one per digit; the brightest one is the network's answer
- Activations in one layer **determine** the activations in the next — that flow of numbers is the network "running."

## How one neuron computes its activation
- Each neuron takes a **weighted sum** of all activations in the previous layer, adds a **bias**, then squishes the result into 0–1:

  a = σ(w₁a₁ + w₂a₂ + … + wₙaₙ + b)

- **Weights** = what pixel pattern this neuron responds to (positive weights excite it, negative weights inhibit it).
- **Bias** = a threshold: how high the weighted sum must be before the neuron meaningfully activates.
- The **sigmoid** σ squishes any number into (0, 1). (Modern networks mostly use **ReLU** instead — mentioned at the end of the video.)

## The hoped-for layered structure
- The *hope* is that layers build up abstraction: pixels → **edges** → **patterns like loops and lines** → **digits** (a 9 = a loop on top + a line).
- Caveat the video itself makes: this is the motivation, not a guarantee — real trained networks don't necessarily learn such clean, human-interpretable pieces.

## The numbers to remember
- This small network has **~13,000 parameters** (13,002 exactly: 784·16 + 16·16 + 16·10 weights, plus 16+16+10 biases).
- **"Learning" = finding good values for all 13,002 knobs.** That's what the gradient descent video (ch. 2) is about.

## Notation worth remembering
- The whole layer-to-layer step compresses to **a⁽¹⁾ = σ(W a⁽⁰⁾ + b)** — a matrix of weights times the activation vector, plus a bias vector, squished. This matrix form is how everything is written and computed from here on.

## Three lines to remember
1. A network is a function built from layers of neurons that each hold a 0–1 number.
2. Each activation = sigmoid(weighted sum of previous layer's activations + bias).
3. The weights and biases are the ~13k dials; training means tuning them.
