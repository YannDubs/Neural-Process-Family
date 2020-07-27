# Latent NPFs



## Overview
In the previous chapter we discussed about CNPFs and argued that although veru useful for arg maximizing the posterior predictive it cannot model arbitrarily complex posterior predictives nor can it be used to sample coherent functions from the posterior predictive. 
Here we will show how to tackle these issues by treating the representation as a latent variable.

The three previously discussed CNPFs can have a corresponding latent variable model (LNPF)[^LNPs], essentially instead of generating a deterministic representation of the context set, the encoder parametrizes a distribution over representations from which we will sample to give rise to different coherent sampled functions from the posterior predictive. By marginalizing over the latent varable LNPFs can also model the arbitrarily complex posterior predictive

* equations
* more intuition

## Training

For training it is the same discussion as usual latent variable models 
1. ideally max likelihood ({doc}`Additional theory <Theory>`). Sampling is possible due to reparametrixation trick [cite], but still computationnaly intractable due to the marginalization so use monte carlo estimation. Problem:  biased, requires many samples since we are in a regime similar to prior sampling ({doc}`Additional theory <Theory>`).
2. An alternative route is to take a VI-inspired approach. Here, we move to "posterior sampling", so may expect to work better with fewer samples. However (i) the interpretation of the approximate posterior is quite troublesome (({doc}`Additional theory <Theory>`), and in fact we often don't care about the latent distribution. (ii) This can be viewed as an odd regularization procedure on ML training, and is not clearly justified. [...equation and approx...]

In the end, determining the "better" approach may be task-dependent, and require empirical evaluation.

```{admonition} Details
---
class: tip
---
Model details in {doc}`Additional Training <Training>`)
```

```{note} 
If you are familiar with the literature on VAE, the disctinction is essentially the same as training with ELBO or IWAE [...]
With the additional issue that the conditional prior cannot is unkown so we replace it with the variational distribution => not proper lower bound any more
```

## Latent Neural Process (LNP)

```{figure} ../images/graph_model_LNPs.svg
---
width: 200em
name: graph_model_LNPs_text
alt: graphical model LNP
---
Graphical model for LNPs.
```
```{figure} ../images/computational_graph_LNPs.svg
---
width: 300em
name: computational_graph_LNPs_text
alt: Computational graph LNP
---
Computational graph for LNPS. [drop?]
```

The latent neural process {cite}`garnelo2018neural` is the latent counterpart of CNP, i.e. once we have a representatoin $R$ we will pass it through a MLP to predict the mean and variance of the latent representation from which to sample.
See {numref}`graph_model_LNPs_text` for the graphical model and {numref}`computational_graph_LNPs_text` for the computational graph [... worth putting  computational graph ? ...].
[theory ...]

```{figure} ../gifs/LNP_rbf.gif
---
width: 35em
name: LNP_rbf_text
alt: LNP on GP with RBF kernel
---
Samples from posterior predictive of LNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`graph_model_LNPs_text` shows that the latent variable indeed enables coherent sampling from the posterior predictive.
It nevertheless suffers from the same underfitting issue as discussed with CNPs.



```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`LNP Notebook <../reproducibility/LNP>`
```

## Attentive Latent Neural Process (AttnLNP)

```{figure} ../images/graph_model_AttnLNPs.svg
---
width: 200em
name: graph_model_AttnLNPs_text
alt: graphical model AttnLNP
---
Graphical model for AttnLNPs.
```
```{figure} ../images/computational_graph_AttnLNPs.svg
---
width: 400em
name: computational_graph_AttnLNPs_text
alt: Computational graph AttnLNP
---
Computational graph for AttnLNPS. [drop?]
```

The Attentive LNPs {cite}`kim2019attentive` is the latent counterpart of AttnCNPs. The way they incorporated the latent variable is a little different than other LNPs, in that they added a "latent path" in addition (not instead) of the deterministic path, giving rise to the (strange?) graphical model depicted in {numref}`graph_model_AttnLNPs_text`.
The latent path is implemented with the same method as LNPs, i.e. a mean aggregation followed by a parametrization of a Gaussian.
In other words, even though the deterministic representation is $R^{(t)}$ is target specific, the latent representation $\mathrm{Z}$  is target independent as seen in the computational graph ({numref}`computational_graph_AttnLNPs_text`).


```{figure} ../gifs/AttnLNP_single_gp.gif
---
width: 35em
name: AttnLNP_single_gp_text
alt: AttnLNP on single GP
---

Samples Posterior predictive of AttnLNPs (Blue) and the oracle GP (Green) with RBF,periodic, and noisy Matern kernel. 
```

From {numref}`AttnLNP_single_gp_text` we see that although the marginal posterior predictive seem good the samples:
1. are not very smooth (the "kinks" seed in {numref}`AttnCNP_single_gp_text` are even more obvious when sampling); 
2. lack diversity and seem to be shifted versions of each other. This is probably because having a very expressive deterministic path diminishes the need of a useful latent path.

Let us now look at images.

```{figure} ../gifs/AttnLNP_img.gif
---
width: 45em
name: AttnLNP_img_text
alt: AttnLNP on CelebA, MNIST, ZSMM
---
Samples from posterior predictive of an AttnCNP for CelebA $32\times32$, MNIST, ZSMM.
```

From {numref}`AttnLNP_img_text` we see that AttnLNP generates nice samples shows descent sampling and good performances when the model does not require generalization (CelebA $32\times32$, MNIST) but breaks for ZSMM as it still cannot extrapolate.

```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`AttnLNP Notebook <../reproducibility/AttnLNP>`
```

## Convolutional Latent Neural Process (ConvLNP)

```{figure} ../images/graph_model_ConvLNPs.svg
---
width: 200em
name: graph_model_ConvLNPs_text
alt: graphical model ConvLNP
---
Graphical model for ConvLNPs.
```
```{figure} ../images/computational_graph_ConvLNPs.svg
---
width: 400em
name: computational_graph_ConvLNPs_text
alt: Computational graph ConvLNP
---
Computational graph for ConvLNPS. [simplify ? useful to mention or show induced points ? drop?]
```

The Convolutional LNPs {cite}`foong2020convnp` is the latent counterpart of ConvCNPs. A major difference compared to AttnLNP is that the latent path *replaces* the deterministic, which is done by actually having a latent functional representation (a latent stochastic process) instead of a latent vector valued variable. [intuition ...]

```{note}
An other way of viewing the ConvLNP is that it consists of 2 stacked ConvCNP, the first one models the latent stochastic process. The second one takes as input a sample from the latent stochastic process and models the posterior predictive conditioned on that sample.
```

* Mention that better trained using MLE probably because of the functional KL [better explanation ?] 
* Link to theory
* talk about global representation ?


```{figure} ../gifs/ConvLNP_single_gp_extrap.gif
---
width: 35em
name: ConvLNP_single_gp_extrap_text
alt: ConvLNP on GPs with RBF, periodic, Matern kernel
---

Samples Posterior predictive of AttnLNPs (Blue) and the oracle GP (Green) with RBF,periodic, and noisy Matern kernel. 
```

From {numref}`ConvLNP_single_gp_extrap_text` we see that ConvLNP performs very well and the samples are reminiscent of those from a GP, i.e., with much richer variability compared to {numref}`AttnLNP_single_gp_text`. 

Let us now make the problem harder by having the ConvLNP model a stochastic process whose posterior predictive is non Gaussian. We will do so by having the following underlying generative process: sample one of the 3 previous kernels then sample funvtion. Note that the data generating process is not a GP (when marginalizing over kernel hyperparameters). Theoretically this could still be modeled by a LNPF as the latent variables could model the current kernel hyperparameter. This is where the use of a global representation makes sense.

```{figure} ../gifs/ConvLNP_kernel_gp.gif
---
width: 35em
name: ConvLNP_kernel_gp_text
alt: ConvLNP trained on GPs with RBF,Matern,periodic kernel
---
Similar to {numref}`ConvLNP_single_gp_extrap_text` but the training was performed on all data simultaneously. 
```

From {numref}`ConvLNP_kernel_gp_text` we see that ConvLNP performs quite well in this harder setting. Indeed, it seems to model process using the periodic kernel when the number of context points is small but quickly (around 10 context points) recovers the correct underlying kernel. Note that we plot the posterior predictive of the actual underlying GP but the generating process is highly non Gaussian.

[should we also add the results of {numref}`ConvLNP_vary_gp` to show that not great when large/ uncountable number of kernels?]

```{figure} ../images/ConvLNP_marginal.png
---
width: 20em
name: ConvLNP_marginal_text
alt: Samples from ConvLNP on MNIST and posterior of different pixels
---
Samples form the posterior predictive of ConvCNPs on MNIST (left) and posterior predictive of some pixels (right).
```

As we discussed in  {ref}`the "CNPG" issue section <issues-cnpfs>`, CNP not only could not be used to generate coherent samples but the posterior predictive is also Gaussian.
{numref}`ConvLNP_marginal_text` shows that both of these issues are somewhat alleviated (compare to {numref}`ConvCNP_marginal_text`) [...]
 

Here are more image samples.

```{figure} ../gifs/ConvLNP_img.gif
---
width: 45em
name: ConvLNP_img_text
alt: ConvLNP on CelebA, MNIST, ZSMM
---
Samples from posterior predictive of an ConvCNP for CelebA $32\times32$, MNIST, ZSMM.
```

[REVERT PLOTS OF CONVLNP, i made a small modifiviation which looks bad...]

### Issues of LNPFs

* Do not think that LNPF are necessarily better than CNP
* Cannot easily arg maximize posterior
* More variance when training
* More computationaly demanding to estimate (marginal) posterior predictive

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.