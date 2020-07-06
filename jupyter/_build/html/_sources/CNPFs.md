# Conditional NPFs

```{figure} images/graph_model_CNPF.svg
---
height: 250px
name: graph_model_CNPF
---
Graphical model for the Conditional Neural Process Sub-Family.
```

Conditional Neural Process Sub-Family (CNPFs), are NPFs that do not have any latent random variables.
They all have the same probabilistic graphical model ({numref}`graph_model_CNPF`), but the computation graph is quite different because of differences in the aggregator $\mathrm{Agg}$.
We will be talking about the following and experimenting with the following:


* {doc}`Conditional Neural Process <CNP>`(CNP) {cite}`garnelo2018conditional`. The aggregator is a simple average.
* {doc}`Attentive CNP <AttnCNP>` (AttnCNP){cite}`kim2019attentive`. The aggregator is an attention mechanism.
* {doc}`Convolutional CNP <ConvCNP>` (ConvCNP) {cite}`gordon2019convolutional`.  The aggregator is a set convolution.

The training of CNPFs is straight forward and consists in optimizing a Monte Carlo estimate of the likelihood (Eq. {eq}`training`).