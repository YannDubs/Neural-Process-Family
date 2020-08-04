# Latent NPFs

## Overview

We concluded the previous section by noting two important drawbacks of the CNPF:

* The marginal predictive distribution is factorised, and thus can neither account for correlations in the predictive nor (as a result) produce "coherent" samples from the predictive distribution.
* The marginal predictive distributions require specification of a particular parametric form.

In this section we discuss an alternative parametrisation of $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$ that still enforces consistency in the predictive, and addresses both of these issues.
The main idea is to introduce a latent variable $\mathbf{z}$ in to the definition of the predictive distribution.
This leads us to the second major branch of the NPF, which we refer to as the Latent Neural Process Sub-family, or LNPF for short.
A graphical representation of the LNPF  is given in {numref}`graph_model_LNPs_text`.

```{figure} ../images/graph_model_LNPs.svg
---
width: 200em
name: graph_model_LNPs_text
alt: graphical model LNP
---
Graphical model for LNPs.
```

To specify this family of models, we must define a few components:

* An encoder: $p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)$, which provides a distribution over the latent variable $\mathbf{z}$ having observed the context set $\mathcal{C}$.
* A decoder: $p_{\boldsymbol\theta} \left( y | x, \mathbf{z} \right)$, which provides predictive distributions conditioned on $\mathbf{z}$ and a target location $x$.

The design of the encoder will follow the principles of the NPF, i.e., using local encodings and a permutation invariant aggregation function.
However, here we will use these principles to model a conditional distribution over the latent variable, rather than a deterministic representation.
A typical example is to have our encoder output the mean and (log) standard deviations of a Gaussian distribution over $\mathbf{z}$.

Again, a typical choice for the decoder is as a Gaussian distribution.
However, as we discuss below, choosing the decoder to have a Gaussian form in this case is far less restrictive than with the CNPF.
With the above components specified, we can now express the predictive distribution as

```{math}
:label: latent_likelihood
\begin{align}
p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \int p_{\boldsymbol\theta} \left(\mathbf{y}_\mathcal{T} , \mathbf{z} | \mathbf{x}_\mathcal{T} , \mathcal{C} \right) \mathrm{d}\mathbf{z} & \text{Parameterisation}  \\
&= \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, \mathbf{z}) \mathrm{d}\mathbf{z}  & \text{Factorisation}\\
&= \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)  \prod_{t=1}^{T} \mathcal{N} \left( y^{(t)};  \mu^{(t)}, \sigma^{2(t)} \right) \mathrm{d}\mathbf{z} & \text{Gaussianity}
\end{align}
```

Now, you might note that we have still made both the factorisation and Gaussian  assumptions!
However, while the decoder likelihood $p_{\boldsymbol\theta}(\mathbf{y} | \mathbf{x}, \mathbf{z})$ is still factorised, the predictive distribution we are actually interested in --- $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$ --- which is defined by _marginalising_ the latent variable $\mathbf{z}$, is no longer factorised, thus addressing the first problem we associated with the CNPF.
Moreover, that distribution, which we refer to as the _marginal predictive_, is no longer Gaussian either.
In fact, by noting that the marginal predictive now has the form of an _infinite mixture of Gaussians_, we can conclude that _any_ predictive distribution can be represented (i.e. learned) by this form.
This is great news, as it (conceptually) relieves us of the burden of choosing / designing an appropriate likelihood when deploying the NPF for a new application!

While this parameterisation seems to solve the major problems associated with the CNPF, it introduces an important drawback.
In particular, the key difficulty with the LNPF is that the likelihood we defined in {numref}`latent_likelihood` is no longer _tractable_.
This has several severe implications, and in particular means that we can longer use simple maximum-likelihood training for the parameters of the model.
In the remainder of this section, we first discuss the  question of training members of the LNPF -- without having any particular member in mind.
After discussing several training procedures, we discuss extensions of each of the CNPF members introduced in the previous section to their corresponding member of the LNPF.

## Training LNPF members

Ideally, what we would like to do is use the likelihood defined in {numref}`latent_likelihood` to optimise the parameters of the model.
However, this quantity is not tractable, and so we must consider alternative procedures.
In fact, this story is not new, and the same issues arise when considering many latent variable models, such as variational auto-encoders (VAEs).

The question of training LNPF members is still open, and there is ongoing research in this area.
In this section, we will cover two methods for training LNPF models, but we emphasise that both have their flaws, and deciding on an appropriate training method is an open question that must often be answered empirically.

```{admonition} A note on chronology
---
class: caution, dropdown
---
In this tutorial, the chronology with which we introduce the objective functions does not follow the order in which they were originally introduced in the literature.
We begin with an approximate maximum-likelihood based procedure, which was recently introduced by {cite}`foong2020convnp` due to its simplicity, and how it relates to the training procedure used for the CNPF.
Following this, we introduce the variational inference inspired approach, which was proposed earlier by {cite}`garnelo2018neural` to train members of the LNPF.
```


### Neural Process Maximum Likelihood (NPML)

First, let's consider a direct approach to optimising the log-marginal predictive likelihood of LNPF members.
While this quantity is no longer tractable (as it was with members of the CNPF), we can derive an estimator of it using Monte-Carlo sampling and the so-called _LogSumExp_ trick:

```{math}
:label: npml
\begin{align}
\log p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \log \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Parameterisation} \\
& \approx \log \left( \frac{1}{L} \sum_{l=1}^{L} \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) & \text{Monte-Carlo approximation} \\
& = \log \left( \sum_{l=1}^{L} \exp \left(  \sum_{t=1}^{T} \log p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) \right) - \log L & \text{LogSumExp trick}\\
& = \text{LogSumExp}_{l=1}^{L} \left( \sum_{t=1}^{T} \log p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) - \log L.
\end{align}
```

{numref}`npml` provides a simple-to-compute objective function for training LNPF-members, which we can then use with standard optimisers to maximise $\hat{L}$ with respect to $\boldsymbol\theta$:
1. Take $L$ samples from the encoder: $\mathbf{z}_l \sim p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)$.
2. For each sample, compute the log-likelihood of the target set: $\log p_l \leftarrow \sum_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z}_l  \right)$.
3. Compute the LogSumExp over the sampling dimension: $\hat{\mathcal{L}}(\boldsymbol\theta; \mathcal{C}, \mathcal{T}) \leftarrow \text{LogSumExp}_{l=1}^{L} \log p_l$

NPML is conceptually very simple.
It also links nicely with the procedure of the CNPF, in the sense that it targets the same predictive likelihood during training.
Moreover, it tends to work well in practice, typically leading to models achieving good performance.
However, it suffers from two important drawbacks:
1. *Bias*: When applying the Monte-Carlo approximation, we have employed an unbiased estimator to the predictive likelihood. However, in practice we are interested in the _log_ likelihood. Unfortunately, the log of an unbiased estimator is not itself unbiased. As a result, NPML is a _biased_ (conservative) estimator of the quantity we would actually like to optimise.
2. *Sample-complexity*: In practice it turns out that NPML is quite sensitive to the number of samples $L$ used to approximate it. In both our GP and image experiments, we find that on the order of 20 samples are required to achieve "good" performance. Of course, the computational and memory costs of training scale with $L$, often limiting the number of samples that can be used in practice.

Unfortunately, addressing the first issue turns out to be quite difficult, and is an open question in training latent variable models in general.
However, we next describe an alternative training procedure, inspired by _variational inference_ that typically works well with fewer samples, often even allowing us to employ single sample estimators in practice.

```{admonition} Further issues
---
class: caution, dropdown
---
Another issue is that the bias in the estimator makes it challenging to exactly carry out quantitative experiments, since our performance metrics are only ever lower bounds to the true model performance, and quantifying how tight those bounds are is quite challenging.
```


### Neural Process Variational Inference (NPVI)

Next, we discuss a training procedure, proposed by {cite}`garnelo2018neural`, which takes inspiration from the literature on variational inference (VI). In particular, we can think about this objective as a variant of _amortised_ VI (used in training VAEs) for the NPF.
There are many resources available on amortised VI (LINKS TO BLOGS / PAPERS), and we encourage readers unfamiliar with the concept to take the time to go through some of these.
Many of the relevant ideas are widely applicable, and provide valuable insights for several sub-areas of ML.
For our purposes, the following intuitions should suffice:

The central idea in amortised VI is to introduce an _inference network_, denoted $q_{\boldsymbol\phi}$, which is trained to approximate the (intractable) _posterior distribution_ over the latent variable.
In our case, the posterior distribution of interest is $p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{C}, \mathcal{T})$, i.e., the distribution of the latent variable having observed _both_ the context and target sets.
To approximate this posterior, we can introduce a network that maps datasets to distributions over the latent variable.
We already know how to define such networks in the NPF -- they require the same form as our encoder $p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{C})$!
In fact, as we discuss below, NPVI proposes to use the encoder as the inference network when training LNPF members.

```{admonition} Inference network details
---
class: dropdown, caution
---
In fact, the inference network is trained to approximate a mapping from the observed data to the posterior distribution over the latent variable.
This is where the term _amortised_ comes from: rather than freely optimise the parameters of each approximate posterior distribution, we share the parameters via a global mapping (often parameterised as a neural network).
```
Having introduced $q_{\boldsymbol\phi}$, we can use it to derive a _lower bound_ (often coined an _ELBO_) to the log marginal likelihood we would like to optimise.
Denoting $\mathcal{D} = \mathcal{C} \cup \mathcal{T}$, we have that

```{math}
:label: np_elbo
\begin{align}
p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
\geq \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol\phi}(\mathbf{z} | \mathcal{D})} \left[ \sum_{t=1}^T \log p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right] - \mathrm{KL} \left( q_{\boldsymbol\phi} \left( \mathbf{z} | \mathcal{D} \right) \| p_{\boldsymbol \theta} \left( \mathbf{z} | \mathcal{C} \right) \right),
\end{align}
```

where $\mathrm{KL}(p \| q)$ is the Kullback-Liebler (KL) divergence between two distributions $p$ and $q$.
We can derive an unbiased estimator to {numref}`np_elbo` by taking samples from $q_{\boldsymbol\phi}$ to estimate the first term on RHS.
When both the encoder and inference network parameterise Gaussian distributions over $\mathbf{z}$ (as is standard), the KL-term can be computed analytically.
Note that {numref}`np_elbo` provides an objective function for training both the inference network parameters $\boldsymbol\phi$ and model parameters $\boldsymbol\theta$.
In fact, typical practice when designing LNPF members is to _share_ the parameters between the encoder and inference network.
When doing so, we can express a single training step for LNPF members with NPVI as follows:

1. Sample a task $(\mathcal{C}, \mathcal{T})$ from the data.
2. Take $L$ samples as $\mathbf{z}_l \sim p_{\boldsymbol\theta} \left(\mathbf{z} | \mathcal{D} \right)$.
3. Approximate the lower-bound as (assuming the KL has an analytical form)
```{math}
\begin{align}
\hat{\mathcal{L}} \leftarrow \frac{1}{L} \sum_{l=1}^{L} \sum_{t=1}^{T} \log p_{\boldsymbol\theta} \left( y^{(t)} |  x^{(t)}, \mathbf{z}_l \right) - \mathrm{KL} \left( p_{\boldsymbol\theta} \left(\mathbf{z} | \mathcal{D} \right) \| p_{\boldsymbol\theta} \left(\mathbf{z} | \mathcal{C} \right) \right).
\end{align}
```
4. Use the backpropagation algorithm to take a gradient step in $\boldsymbol\theta$ to maximize $\hat{\mathcal{L}}$.


```{admonition} Proper derivations of the NPVI ELBO
---
class: caution, dropdown
---
There are several nuances with the derivation of the NPVI ELBO that we have glossed over for the sake of brevity.
In fact, the original derivation of this procedure ({cite}`garnelo2018neural`) invokes slightly different modelling assumptions than what we have used, and ends up introducing an approximation which results in the resulting ELBO not being a proper lower bound on the log-marginal likelihood.
In the {doc}`Theory <Theory>` chapter we discuss in detail the modelling assumptions associated with the derivations, provide a full derivation of the ELBO, and discuss the implications of the approximations that must be made along the way.
```

NPVI inherits several appealing properties from the VI framework:
* It utilises _posterior sampling_ to reduce the variance of the Monte-Carlo estimator of the intractable expectation. This means that often we can get away with training models taking just a single sample, resulting in computationally and memory efficient training procedures.
* If our approximate posterior can recover the true posterior, the inequality is tight, and we are exactly optimising the log-marginal likelihood.

However, it also inherits the main drawbacks of VI.
In particular, it is almost never the case that the true posterior can be recovered by our approximate posterior in any practical application.
In these settings:

* Meaningful guarantees about the quality of the learned models are hard to come by.
* The inequality holds, meaning that we are only optimising a lower-bound to the quantity we actually care about. Moreover, it is often quite difficult to know how tight this bound may be.

```{admonition} Biased estimators
---
class: caution, dropdown
---
Importantly, the second drawback means that NPVI is not actually avoiding the "biased estimator" issue of NPML.
Rather, NPML defines a biased (conservative) estimator of the desired quantity, and NPVI produces an unbiased estimator of a strict lower-bound of the same quantity.
In both cases, sadly, we do not have unbiased estimators of the quantities we really care about.
```

Finally, NPVI adds additional drawbacks that are unique to the NPF setting.
These can be roughly summarised as

* VI is heavily focused on approximating the posterior distribution, and diverts encoder capacity and attention of the optimiser to recovering the true posterior. However, in the NPF setting, we are typically only interested in the predictive distribution $p(\mathbf{y}_T | \mathbf{x}_T, \mathcal{C})$, and it is unclear whether focusing our efforts on $\mathbf{z}$ is beneficial to achieving higher quality predictive distributions.
* Sharing the encoder and inference network introduces additional complexities in the training procedure. In particular, it muddies the distinction between modelling and inference that is typical in probabilistic modelling. As a result, it is unclear what the effect of the dual roles of the encoder in the model are, and it may be that using the encoder as an approximate posterior has a detrimental effect on the resulting predictive distributions.

Despite these (and other) drawbacks NPVI is the most commonly employed procedure for training LNPF members.
Next, we turn our attention to several members of the LNPF.
In particular, we will introduce the latent-variable variant of each of the conditional models introduced in the previous section, and we shall see that having addressed the training procedures, the extension to latent variables is quite straightforward from a practical perspective.
After introducing the models, we will illustrate a brief comparison of the two described training procedures.



## Latent Neural Process (LNP)

```{figure} ../images/computational_graph_LNPs.svg
---
width: 300em
name: computational_graph_LNPs_text
alt: Computational graph LNP
---
Computational graph for LNPS. [drop?]
```

The latent neural process {cite}`garnelo2018neural` is the latent counterpart of the CNP, and the first member of the LNPF proposed in the literature.
Given the vector $R$, which is computed as in the CNP, we simply pass it through an additional MLP to predict the mean and variance of the latent representation $\mathbf{z}$, from which we can produce samples.
We call the computational graph that produces the distribution over $\mathbf{z}$ the _latent path_, to distinguish from a computation graph for a deterministic representation, which we refer to as a _deterministic path_.
The decoder then has the same architecture, and we can simply pass samples of $\mathbf{z}$, together with desired target locations, to produce our predictive distributions.
We typically treat samples from the predictive as the mean functions produced by passing one sample of $\mathbf{z}$ through the decoder.
{numref}`computational_graph_LNPs_text` illustrates the computational graph of the LNP.

```{note} Parameterising the observation noise
---
class: tip, dropdown
---
A particularly important detail in the parameterisation of LNPF-members is how the predictive standard deviation is handled.
Recall that for CNPF members, our decoder parameterised both a mean and (log) standard deviation for each target location.
However, now we have an additional source of uncertainty, which arises from the uncertainty in the latent variable.
For example, consider taking $L$ samples from the latent variable, and using each to make a prediction for a particular target location $x^{(t)}$.
The uncertainty in the prediction of $y^{(t)}$ would also be influenced by the variance in the _means_ $\mu^{(t)_l}, which would be different for every target location.

Thus, we could model _heteroskedastic_ noise via the latent variable, and employ a global (learned) observation noise parameter, with the decoder only outputting the mean for each prediction.
However, in practice we find (as have several other papers dealing with the LNPF) that continuing to model the observation noise as an output of the decoder tends to lead to better performance in most cases.
Thus, throughout our experiments we employ what we call _heteroskedastic observation noise_, which is just a fancy way of saying that the decoder outputs both a mean and (log) standard deviation.

YANN: DO YOU WANT TO ADD A NOTE HERE ABOUT HOW THIS INTERACTS WITH LOWER BOUNDING CERTAIN QUANTITIES?
```

Below, we show the predictive distribution of an LNP trained on samples from the RBF-kernel GP, as the number of observed context points from the underlying function increases.
Note that here this model is trained with NPVI. (SHOULD WE PUT IT SIDE BY SIDE WITH A MODEL TRAINED WITH NPML?)


```{figure} ../gifs/LNP_rbf.gif
---
width: 35em
name: LNP_rbf_text
alt: LNP on GP with RBF kernel
---
Samples from posterior predictive of LNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`graph_model_LNPs_text` shows that the latent variable indeed enables coherent sampling from the posterior predictive.
In fact, within the range $(-1, 1)$ the model produces very nice samples, that seem to properly mimic those from the underlying process.
Nevertheless, here too we see that the LNP suffers from the same underfitting issue as discussed with CNPs.
We again see the tendency to overestimate the uncertainty, and often not pass through all the context points with the mean functions.
Moreover, we can observe that beyond the $(-1, 1)$, the model seems to "give up" on the context points and uncertainty, despite having been trained on the range $(-2, 2)$.



```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`LNP Notebook <../reproducibility/LNP>`
```



## Attentive Latent Neural Process (AttnLNP)


The Attentive LNPs {cite}`kim2019attentive` is the latent counterpart of AttnCNPs.
Differently from the LNP, the AttnLNP _added_ a "latent path" in addition to (rather than instead of) the deterministic path, giving rise to the slightly different graphical model depicted in {numref}`graph_model_AttnLNPs_text`.

```{figure} ../images/graph_model_AttnLNPs.svg
---
width: 200em
name: graph_model_AttnLNPs_text
alt: graphical model AttnLNP
---
Graphical model for AttnLNPs.
```

The latent path is implemented with the same method as LNPs, i.e. a mean aggregation followed by a parametrization of a Gaussian.
In other words, even though the deterministic representation is $R^{(t)}$ is target specific, the latent representation $\mathrm{Z}$  is target independent as seen in the computational graph ({numref}`computational_graph_AttnLNPs_text`).

```{figure} ../images/computational_graph_AttnLNPs.svg
---
width: 400em
name: computational_graph_AttnLNPs_text
alt: Computational graph AttnLNP
---
Computational graph for AttnLNPS. [drop?]
```

Below, we show the predictive distribution of an AttnLNP trained on samples from RBF, periodic, and noisy Matern kernel GPs, again viewing the predictive as the number of observed context points is increased.
Here too, the model is trained with NPVI. (SHOULD WE PUT THESE SIDE BY SIDE WITH A MODEL TRAINED WITH NPML? MIGHT BE A LITTLE MUCH HERE, BUT I THINK IT COULD BE HELPFUL TO HAVE THESE COMPARISONS.)

```{figure} ../gifs/AttnLNP_single_gp.gif
---
width: 35em
name: AttnLNP_single_gp_text
alt: AttnLNP on single GP
---

Samples Posterior predictive of AttnLNPs (Blue) and the oracle GP (Green) with RBF,periodic, and noisy Matern kernel.
```

{numref}`AttnLNP_single_gp_text` paints an interesting picture regarding the AttnLNP.
On the one hand, we see that it is able to do a significantly better job in modelling the marginals than the LNP.
However, on closer inspection, we can see several issues with the resulting distributions:
1. The samples do not seem to be smooth, and we see "kinks" that are similar (though even more pronounced) than in {numref}`AttnCNP_single_gp_text`.
2. Moreover, in many places the AttnLNP seems to collpase the distribution around the latent variable, and express all its uncertainty via the observation noise. We coin this behaviour "collapsing onto its conditional variant", and note that this tends to occur more often for the AttnLNP when trained with NPVI rather than NPML.
Let us now consider the image experiments as we did with the AttnCNP.

```{figure} ../gifs/AttnLNP_img.gif
---
width: 45em
name: AttnLNP_img_text
alt: AttnLNP on CelebA, MNIST, ZSMM
---
Samples from posterior predictive of an AttnCNP for CelebA $32\times32$, MNIST, ZSMM.
```

From {numref}`AttnLNP_img_text` we see that AttnLNP generates quite impressive samples, and exhibits descent sampling and good performances when the model does not require generalisation (CelebA $32\times32$, MNIST).
However, as expected, the model "breaks" for ZSMM as it still cannot extrapolate.

```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`AttnLNP Notebook <../reproducibility/AttnLNP>`
```


## Convolutional Latent Neural Process (ConvLNP)

The Convolutional LNP {cite}`foong2020convnp` is the latent counterpart of the ConvCNP.
Similar to the LNP (and in contrast with the AttnLNP), the latent path _replaces_ the deterministic one, resulting in a latent functional representation (a latent stochastic process) instead of a latent vector valued variable.
We can represent this model with a graphical representation as illustrated in {numref}`graph_model_ConvLNPs_text`.

```{figure} ../images/graph_model_ConvLNPs.svg
---
width: 200em
name: graph_model_ConvLNPs_text
alt: graphical model ConvLNP
---
Graphical model for ConvLNPs.
```

Another way of viewing the ConvLNP, which is useful in gaining an intuitive understanding of the computational graph (see {numref}`computational_graph_ConvLNPs_text`) is as consisting of two stacked ConvCNPs: the first takes in context sets and outputs a latent stochastic process (the encoder).
The second takes as input a sample from the latent process and models the posterior predictive conditioned on that sample (the decoder).

```{figure} ../images/computational_graph_ConvLNPs.svg
---
width: 400em
name: computational_graph_ConvLNPs_text
alt: Computational graph ConvLNP
---
Computational graph for ConvLNPS. [simplify ? useful to mention or show induced points ? drop?]
```

One difficulty arises in training the ConvLNP with the NPVI objective, as it requires evaluating the KL divergence between two stochastic processes, which is a tricky proposition.
{cite}`foong2020convnp` propose a simple approach, that approximates this quantity by instead summing the KL divergences at each discretisation location.
However, as they note, the ConvLNP tends to perform significantly better in most cases when trained with NPML rather than NPVI.
Below, we show similar plots for the ConvLNP on the GP experiments.
However, here we are illustrating the performance of a ConvLNP trained with NPML, not NPVI. (SAME QUESTION: SHOULD WE PUT THESE SIDE BY SIDE WITH A MODEL TRAINED WITH NPML?.)

```{note} Global representation ConvLNPs
---
class: tip, dropdown
---
In this tutorial, we consider a simple extension to the ConvLNP proposed by {cite}`foong2020convnp`, which includes a _global latent representation_ as well.
The global representation is computed by average-pooling half of the channels in the latent function, resulting in a translation _invariant_ latent representation (further details regarding this can be found in the ConvLNP notebook).
The intuition behind such a representation is that it may help to capture aspects of the underlying function that are global, allowing the functional representation to represent more localised information.
This intuition is clearest when considering mixture of GPs experiments discussed below.
```

```{figure} ../gifs/ConvLNP_single_gp_extrap.gif
---
width: 35em
name: ConvLNP_single_gp_extrap_text
alt: ConvLNP on GPs with RBF, periodic, Matern kernel
---
Samples Posterior predictive of ConvLNPs (Blue) and the oracle GP (Green) with RBF,periodic, and noisy Matern kernel.
```

From {numref}`ConvLNP_single_gp_extrap_text` we see that ConvLNP performs very well and the samples are reminiscent of those from a GP, i.e., with much richer variability compared to {numref}`AttnLNP_single_gp_text`.
Further, as in the case of the ConvCNP, we see that the ConvLNP elegantly generalises beyond the range in $X$-space on which it was trained.

Next, we consider the more challenging problem of having the ConvLNP model a stochastic process whose posterior predictive is non Gaussian.
We do so by having the following underlying generative process: first, sample one of the 3 kernels discussed above, and second, sample a function from the sampled kernel.
Importantly, the data generating process is a mixture of GPs, and the true posterior predictive process (achieved by marginalising over kernel hyperparameters) does not have a closed form representation.
Theoretically, this could still be modelled by a LNPF as the latent variables could represent the current kernel hyperparameters as well.


```{figure} ../gifs/ConvLNP_kernel_gp.gif
---
width: 35em
name: ConvLNP_kernel_gp_text
alt: ConvLNP trained on GPs with RBF,Matern,periodic kernel
---
Similar to {numref}`ConvLNP_single_gp_extrap_text` but the training was performed on all data simultaneously.
```

{numref}`ConvLNP_kernel_gp_text` demonstrates that ConvLNP performs quite well in this harder setting. Indeed, it seems to model process using the periodic kernel when the number of context points is small but quickly (around 10 context points) recovers the correct underlying kernel. Note that we plot the posterior predictive of the actual underlying GP but the generating process is highly non Gaussian.

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

[REVERT PLOTS OF CONVLNP, I made a small modificiation which looks bad...]

### Issues of LNPFs

* Do not think that LNPF are necessarily better than CNP
* Cannot easily arg maximize posterior
* More variance when training
* More computationaly demanding to estimate (marginal) posterior predictive

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.
