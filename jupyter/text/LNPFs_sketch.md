# Latent NPFs

## Overview

We concluded the previous section by noting two important drawbacks of the CNPF:

* The marginal predictive distribution is factorised, and thus can neither account for correlations in the predictive nor (as a result) produce "coherent" samples from the predictive distribution.
* The marginal predictive distributions require specification of a particular parametric form.

In this section we discuss an alternative parametrisation of $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$ that still enforces consistency in the predictive, and addresses both of these issues.
The main idea is to introduce a latent variable $\mathbf{z}$ into the definition of the predictive distribution.
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

* An encoder: $p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)$, which provides a _distribution_ over the latent variable $\mathbf{z}$ having observed the context set $\mathcal{C}$.
* A decoder: $p_{\boldsymbol\theta} \left( y | x, \mathbf{z} \right)$, which provides predictive distributions conditioned on $\mathbf{z}$ and a target location $x$.

The design of the encoder will follow the principles of the NPF, i.e., using local encodings and a permutation invariant aggregation function.
However, here we will use these principles to model a conditional distribution over the latent variable, rather than a deterministic representation.
A typical example is to have our encoder output the mean and (log) standard deviations of a Gaussian distribution over $\mathbf{z}$.

Our decoder accepts as inputs instantiations (typically _samples_) of $\mathbf{z}$ and target locations $x$, and outputs a predictive distribution over target values.
Here too, a typical choice for the decoder is as a Gaussian distribution.
However, as we discuss below, choosing the decoder to have a Gaussian form in this case is far less restrictive than with the CNPF.
With the above components specified, we can now express the predictive distribution as

```{math}
:label: latent_likelihood
\begin{align}
p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \int p_{\boldsymbol\theta} \left(\mathbf{y}_\mathcal{T} , \mathbf{z} | \mathbf{x}_\mathcal{T} , \mathcal{C} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation}  \\
&= \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, \mathbf{z}) \, \mathrm{d}\mathbf{z}  & \text{Factorisation}\\
&= \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)  \prod_{t=1}^{T} \mathcal{N} \left( y^{(t)};  \mu^{(t)}, \sigma^{2(t)} \right) \mathrm{d}\mathbf{z} & \text{Gaussianity}
\end{align}
```

```{admonition} Advanced$\qquad$Latent variable $\implies$ consistency
---
class: hint, dropdown
---
We show that members of the LNPF also specify consistent stochastic processes conditioned on a fixed context set $\mathcal{C}$. To see that the predictive distributions are consistent under permutation, , let $\mathbf{x}_{\mathcal{T}} = \{ x^{(t)} \}_{t=1}^T$ be target inputs. Let $\pi$ be a permutation of $\{1, ..., T\}$. Then the predictive density is (suppressing the $\mathcal{C}$-dependence):

$$
\begin{align}
    p_\theta(y^{(1)}, ..., y^{(T)} | x^{(1)}, ..., x^{(T)}) &= \int p_{\boldsymbol\theta} ( \mathbf{z}) \prod_{t=1}^{T} p_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, \mathbf{z}) \, \mathrm{d}\mathbf{z} \\
    &= p_\theta(y^{(\pi(1))}, ..., y^{(\pi(T))} | x^{(\pi(1))}, ..., x^{(\pi(T))}),
\end{align}
$$

again because multiplication is commutative. To show consistency under marginalisation, we again consider a pair of target inputs, $x^{(1)}, x^{(2)}$. By marginalising out the second target output, we get:

$$
\begin{align}
    \int p_\theta(y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}, \mathcal{C}) \, \mathrm{d}y^{(2)} &= \int \int p_\theta(y^{(1)}| x^{(1)}, \mathbf{z})p_\theta(y^{(2)}| x^{(2)}, \mathbf{z}) p_\theta(\mathbf{z}) \, \mathrm{d}\mathbf{z} \mathrm{d}y^{(2)} \\
    &= \int  p_\theta(y^{(1)}| x^{(1)}, \mathbf{z}) p_\theta(\mathbf{z}) \int p_\theta(y^{(2)}| x^{(2)}, \mathbf{z})  \, \mathrm{d}y^{(2)} \mathrm{d}\mathbf{z}  \\
    &= \int  p_\theta(y^{(1)}| x^{(1)}, \mathbf{z}) p_\theta(\mathbf{z}) \mathrm{d}\mathbf{z}  \\
    &= p_\theta(y^{(1)}| x^{(1)} \mathcal{C}).
\end{align}
$$

which shows that the predictive distribution obtained by querying an LNPF member at $x^{(1)}$ is the same as that obtained by querying it at $x^{(1)}, x^{(2)}$ and then marginalising out the second target point. Of course, the same idea works with collections of any size, and marginalising any subset of the variables.
```

Now, you might be worried that we have still made both the factorisation and Gaussian assumptions!
However, while the decoder likelihood $p_{\boldsymbol\theta}(\mathbf{y} | \mathbf{x}, \mathbf{z})$ is still factorised, the predictive distribution we are actually interested in --- $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$ --- which is defined by _marginalising_ the latent variable $\mathbf{z}$, is no longer factorised, thus addressing the first problem we associated with the CNPF.
Moreover, that distribution, which we refer to as the _marginal predictive_, is no longer Gaussian either.
In fact, by noting that the marginal predictive now has the form of an _infinite mixture of Gaussians_, we can conclude that _any_ predictive distribution can be represented (i.e. learned) by this form.
This is great news, as it (conceptually) relieves us of the burden of choosing / designing an appropriate likelihood when deploying the NPF for a new application!

While this parameterisation seems to solve the major problems associated with the CNPF, it introduces an important drawback.
In particular, the key difficulty with the LNPF is that the likelihood we defined in {numref}`latent_likelihood` is no longer _tractable_.
This has several severe implications, and in particular means that we can longer use simple maximum-likelihood training for the parameters of the model.
In the remainder of this section, we first discuss the  question of training members of the LNPF, without having any particular member in mind.
After discussing several training procedures, we introduce extensions of each of the CNPF members discussed in the previous section to their corresponding member of the LNPF.

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
While this quantity is no longer tractable (as it was with members of the CNPF), we can derive an estimator using Monte-Carlo sampling and the so-called _LogSumExp_ trick:

```{math}
:label: npml
\begin{align}
\log p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \log \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation} \\
& \approx \log \left( \frac{1}{L} \sum_{l=1}^{L} \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) & \text{Monte-Carlo approximation} \\
& = \log \left( \sum_{l=1}^{L} \exp \left(  \sum_{t=1}^{T} \log p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) \right) - \log L & \text{LogSumExp trick}\\
& = \text{LogSumExp}_{l=1}^{L} \left( \sum_{t=1}^{T} \log p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \right) - \log L.
\end{align}
```

{numref}`npml` provides a simple-to-compute objective function for training LNPF-members, which we can then use with standard optimisers to train the model parameters $\boldsymbol\theta$.
Pseudo-code for such a training step is given in {numref}`npml_pseudocode`.

```{figure} ../images/alg_npml.png
---
width: 35em
name: npml_pseudocode
alt: Pseudo-code for a single training step of a LNPF member with NPML.
---
```

NPML is conceptually very simple.
It also links nicely with the procedure of the CNPF, in the sense that it targets the same predictive likelihood during training.
Moreover, it tends to work well in practice, typically leading to models achieving good performance.
However, it suffers from two important drawbacks:
1. *Bias*: When applying the Monte-Carlo approximation, we have employed an unbiased estimator to the predictive likelihood. However, in practice we are interested in the _log_ likelihood. Unfortunately, the log of an unbiased estimator is not itself unbiased. As a result, NPML is a _biased_ (conservative) estimator of the quantity we would actually like to optimise.
2. *Sample-complexity*: In practice it turns out that NPML is quite sensitive to the number of samples $L$ used to approximate it. In both our GP and image experiments, we find that on the order of 20 samples are required to achieve "good" performance. Of course, the computational and memory costs of training scale with $L$, often limiting the number of samples that can be used in practice.

Unfortunately, addressing the first issue turns out to be quite difficult, and is an open question in training latent variable models in general.
However, we next describe an alternative training procedure, inspired by _variational inference_ that typically works well with fewer samples, often even allowing us to employ single sample estimators in practice.


### Neural Process Variational Inference (NPVI)

Next, we discuss a training procedure proposed by {cite}`garnelo2018neural`, which takes inspiration from the literature on variational inference (VI).
The central idea behind this objective function is to use _posterior sampling_ to reduce the sample complexity issue associated with NPML.
For a better intuition regarding this, note that NPML is defined using an expectation against $p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{C})$.
The idea in posterior sampling is to use all the data available during training to produce the distribution over $\mathbf{z}$, thus leading to more informative samples and lower variance objectives.

In our case, the posterior distribution from which we would like to sample is $p(\mathbf{z} | \mathcal{C}, \mathcal{T})$, i.e., the distribution of the latent variable having observed _both_ the context and target sets.
Unfortunately, this posterior is intractable.
To address this, {cite}`garnelo2018neural` propose to replace the true posterior with simply passing both the context and target sets through the encoder, i.e.

```{math}
---
label: approximate_posterior
---
\begin{align}
p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) \approx p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \cup \mathcal{T} \right).
\end{align}
```


```{admonition} Warning$\qquad$Posterior vs. $p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \cup \mathcal{T} \right)$
---
class: warning
---
Note that these two distributions are different.
We can compute $p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \cup \mathcal{T} \right)$ by simply passing both $\mathcal{C}$ and $\mathcal{T}$ through the model encoder.
One the other hand, the posterior $p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right)$ is computed using Bayes' rule and our model definition:

$$
\begin{align}
p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) = \frac{p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right)}{ \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \mathrm{d} \mathbf{z}}.
\end{align}
$$

Recalling that our decoder is defined by a complicated, non-linear neural network, we can see that this posterior is intractable, as it involves an integration against complicated likelihoods.
```

In fact, we can use such an approximation to derive a _lower bound_ to the log marginal likelihood.
The key insight in deriving this bound is to (i) introduce the approximate posterior as a sampling distribution, and (ii) employ a straightforward application of [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality).
Denoting $\mathcal{D} = \mathcal{C} \cup \mathcal{T}$, we have that

```{math}
:label: npvi
\begin{align}
\log p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \log \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)  p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation} \\
& = \log \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{D} \right) \frac{p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)}{p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{D} \right)} p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Approximate posterior} \\
& \geq \int p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{D} \right) \left( \log p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) + \log \frac{p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right)}{p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{D} \right)} \right) & \text{Jensen's inequality} \\
& = \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol\phi}(\mathbf{z} | \mathcal{D})} \left[ \log p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) \right] - \mathrm{KL} \left( q_{\boldsymbol\phi} \left( \mathbf{z} | \mathcal{D} \right) \| p_{\boldsymbol \theta} \left( \mathbf{z} | \mathcal{C} \right) \right),
\end{align}
```

where $\mathrm{KL}(p \| q)$ is the Kullback-Liebler (KL) divergence between two distributions $p$ and $q$, and we have used the shorthand $p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) = \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right)$ to ease notation.
Let's consider what we have achieved in {numref}`npvi`.
We have taken our original objective, and re-expressed it as an expectation with respect to an approximate posterior distribution.
The hope is that, by sampling from this more informative distribution, we can produce more meaningful samples and thus reduce the number of samples we need to properly estimate this objective.

```{admonition} Important
---
class: attention
---
Of course, we can only sample from this approximate posterior during training, when we assume access to both the context _and_ target sets.
At test time, we will only have access to the context set, and so the forward pass through the model will be equivalent to that of the model when trained with NPML, i.e., we will only pass the context set through the encoder.
This is an important detail of NPVI: forward passes at meta-train time look different than they do at meta-test time!
```

There is an important connection between the above procedure and _amortised_ VI.
Amortised VI is a method for performing approximate inference and learning in probabilistic latent variable models, which uses a similar trick to train an _inference_ network to approximate the true posterior distribution under the model.

```{admonition} Advanced$\qquad$LNPF and Amortised VI
---
class: attention, dropdown
---
There are many resources available on (amortised) VI (e.g., [Jaan Altosaar's VAE tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/), [The spectator's](http://blog.shakirm.com/2015/01/variational-inference-tricks-of-the-trade/) take, or this [review paper on modern VI](https://arxiv.org/pdf/1711.05597.pdf)), and we encourage readers unfamiliar with the concept to take the time to go through some of these.
Many of the relevant ideas are widely applicable, and provide valuable insights for several sub-areas of ML.
For our purposes, the following intuitions should suffice:

Assume that we have a latent variable model with a prior $p(\mathbf{z})$, and a conditional likelihood for observations $y$, written $p_{\boldsymbol\theta} \left(y | \mathbf{z} \right)$.
The central idea in amortised VI is to introduce an _inference network_, denoted $q_{\boldsymbol\phi}(\mathbf{z} | y)$, which maps observations to distributions over $\mathbf{z}$.
We can then use $q_{\boldsymbol\phi}$ to derive a lower bound to the log-marginal likelihood, just as we did above:

$$
\begin{align}
\log p_{\boldsymbol\theta}(y)
&= \log \int p_{\boldsymbol\theta} \left( \mathbf{z} , y \right)\mathrm{d}\mathbf{z} \\
& = \log \int q_{\boldsymbol\phi} \left( \mathbf{z} | y \right) \frac{p_{\boldsymbol\theta} \left( \mathbf{z} , y \right)}{q_{\boldsymbol\phi} \left( \mathbf{z} | y \right)} \mathrm{d}\mathbf{z} \\
& \geq \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol\phi} \left( \mathbf{z} | y \right)} \left[ \log p_{\boldsymbol\theta} \left( y | \mathbf{z} \right) \right] - \mathrm{KL} \left( q_{\boldsymbol\phi} \left( \mathbf{z} | y \right) \| p_{\boldsymbol \theta} \left( \mathbf{z} \right) \right).
\end{align}
$$

In the VI terminology, this lower bound is commonly known as the _evidence lower bound_ (ELBO).
So maximising the ELBO with respect to $\boldsymbol\theta$ trains the model to optimise a proper lower bound on the log-marginal likelihood, which is a sensible thing to do.
Moreover, it turns out that that maximising the ELBO with respect to $\boldsymbol\phi$ minimises the KL divergence between the resulting distributions and the true posterior, and so we can think of $q_{\boldsymbol\phi}$ as approximating the true posterior in a meaningful way.

In the NPF, to approximate the desired posterior, we can introduce a network that maps datasets to distributions over the latent variable.
We already know how to define such networks in the NPF -- they require the same form as our encoder $p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{C})$!
In fact, NPVI proposes to use the encoder as the inference network when training LNPF members.

So we can view {numref}`npvi` as performing amortised VI for any member of the LNPF.
The twist on standard amortised VI is that here, we are sharing $q_{\boldsymbol\phi}$ with a part of the model itself, which somewhat complicates our understanding of the procedure, and may lead to unintended consequences.

As such, NPVI inherits several appealing properties from the VI framework:
* It utilises _posterior sampling_ to reduce the variance of the Monte-Carlo estimator of the intractable expectation. This means that often we can get away with training models taking just a single sample, resulting in computationally and memory efficient training procedures.
* If our approximate posterior can recover the true posterior, the inequality in the ELBO is tight, and we are exactly optimising the log-marginal likelihood.

However, it also inherits the main drawbacks of VI.
In particular, it is almost never the case that the true posterior can be recovered by our approximate posterior in any practical application.
In these settings:

* Meaningful guarantees about the quality of the learned models are hard to come by.
* The inequality holds, meaning that we are only optimising a lower-bound to the quantity we actually care about. Moreover, it is often quite difficult to know how tight this bound may be.
```

When both the encoder and inference network parameterise Gaussian distributions over $\mathbf{z}$ (as is standard), the KL-term can be computed analytically.
Hence we can derive an unbiased estimator to {numref}`np_elbo` by taking samples from $q_{\boldsymbol\phi}$ to estimate the first term on the RHS.
{numref}`npvi_pseudocode` provides the pseudo-code for a single training iteration for a LNPF member, using NPVI as the target objective.

```{figure} ../images/alg_npvi.png
---
width: 35em
name: npvi_pseudocode
alt: Pseudo-code for a single training step of a LNPF member with NPML.
---
```

As we have discussed, the most appealing property of NPVI is that it utilises _posterior sampling_ to reduce the variance of the Monte-Carlo estimator of the intractable expectation.
This means that often we can get away with training models taking just a single sample, resulting in computationally and memory efficient training procedures.
However, it also comes with several drawbacks, which can be roughly summarised as follows:

* NPVI is focused on approximating the posterior distribution, and diverts encoder capacity and attention of the optimiser to recovering the true posterior. However, in the NPF setting, we are typically only interested in the predictive distribution $p(\mathbf{y}_T | \mathbf{x}_T, \mathcal{C})$, and it is unclear whether focusing our efforts on $\mathbf{z}$ is beneficial to achieving higher quality predictive distributions.
* In NPVI, the encoder plays a dual role: it is both part of the model, and used as an inference network. This fact introduces additional complexities in the training procedure, and it may be that using the encoder as an approximate posterior has a detrimental effect on the resulting predictive distributions.

As we shall see below, it is often the case that models trained with NPML produce better fits than equivalent models trained with NPVI, at the cost of additional computational and memory costs of training.
In pratice however, NPVI is the most commonly employed procedure for training LNPF members.

```{admonition} Caution$\qquad$Biased estimators
---
class: caution, dropdown
---
Importantly, it is not the case that NPVI avoids the "biased estimator" issue of NPML.
Rather, NPML defines a biased (conservative) estimator of the desired quantity, and NPVI produces an unbiased estimator of a strict lower-bound of the same quantity.
In both cases, sadly, we do not have unbiased estimators of the quantities we really care about.

Another consequence of this is that it is challenging to exactly carry out quantitative experiments, since our performance metrics are only ever lower bounds to the true model performance, and quantifying how tight those bounds are is quite challenging.
```

```{admonition} Advanced$\qquad$Relationship between NPML and NPVI
---
class: attention, dropdown
---
In fact, there is a close relationship between the NPML and NPVI objectives.
To see this, we denote

$$
\begin{align}
  Z = \int p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \mathrm{d}\mathbf{z},
\end{align}
$$

and note that $Z$ is the appropriate _normalising constant_ for the distribution $p_{\boldsymbol\theta} \left( \mathbf{y}_{T}, \mathbf{z} | \mathbf{x}_{T}, \mathcal{C} \right)$.
Now, we can rewrite the NPVI objective as

$$
\begin{align}
\mathcal{L}_{VI} & (\boldsymbol\theta ; D) = \mathbb{E}_{p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{D})} \left[ \log p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) \right] - \mathrm{KL} \left( p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{D} \right) \| p_{\boldsymbol \theta} \left( \mathbf{z} | \mathcal{C} \right) \right) \\
& = \mathbb{E}_{p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{D})} \left[ \log p_{\boldsymbol\theta} \left( \mathbf{y}_{T} | \mathbf{x}_{T}, \mathbf{z} \right) + \log p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) - \log p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \right] \\
& = \mathbb{E}_{p_{\boldsymbol\theta}(\mathbf{z} | \mathcal{D})} \left[ \log Z + \log \frac{1}{Z} p_{\boldsymbol\theta} \left( \mathbf{y}_{T}, \mathbf{z} | \mathbf{x}_{T}, \mathcal{C} \right) - \log p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \right] \\
& = \mathcal{L}_{ML}(\boldsymbol\theta; D) - \mathrm{KL} \left( p_{\boldsymbol\theta} \left( \mathbf{z} | \mathcal{C} \right) \| \frac{1}{Z} p_{\boldsymbol\theta} \left( \mathbf{y}_{T}, \mathbf{z} | \mathbf{x}_{T}, \mathcal{C} \right) \right).
\end{align}
$$

Thus, we can see that $\mathcal{L}_{VI}$ is equal to $\mathcal{L}_{ML}$ up to an additional KL term.
This KL term has a nice interpretation as encouraging consistency among predictions with different context sets, which is the kind of consistency _not_ achieved by the NPF.

However, this term can also be a _distractor_.
When dealing with NPF members, we are typically not interested in the latent variable $\mathbf{z}$, and most considered tasks require only a "good" approximation to the predictive distribution.
Therefore, given only finite capacity of our models, and finite data, it may be preferable to focus all the model capacity on achieving the best possible predictive distribution (which is what $\mathcal{L}_{ML}$ focuses on), rather than focusing on additional properties of $\mathbf{z}$, as encouraged by the KL term introduced by $\mathcal{L}_{VI}$.
```

Armed with procedures for training LNPF-members, we turn our attention to the models themselves.
In particular, we next introduce the latent-variable variant of each of the conditional models introduced in the previous section, and we shall see that having addressed the training procedures, the extension to latent variables is quite straightforward from a practical perspective.



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
The decoder then has the same architecture as that of the CNP, and we can simply pass samples of $\mathbf{z}$, together with desired target locations, to produce our predictive distributions.
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
In other words, even though the deterministic representation $R^{(t)}$ is target specific, the latent representation $\mathbf{z}$  is target independent as seen in the computational graph ({numref}`computational_graph_AttnLNPs_text`).

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

Another way of viewing the ConvLNP, which is useful in gaining an intuitive understanding of the computational graph (see {numref}`computational_graph_ConvLNPs_text`) is as consisting of two stacked ConvCNPs: the first (the encoder) takes in context sets and outputs a latent stochastic process.
The second (the decoder) takes as input a sample from the latent process and models the posterior predictive conditioned on that sample.

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
This intuition is clearest when considering the mixture of GPs experiments discussed below.
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
Importantly, the data generating process is a mixture of GPs, and the true posterior predictive process (achieved by marginalising over the different kernels) does not have a closed form representation.
Theoretically, this could still be modelled by a LNPF as the latent variables could represent the both the current kernel and its functions, but adding the global representation corresponds (intuitively) to allowing the ConvLNP to "switch" between kernels.


```{figure} ../gifs/ConvLNP_kernel_gp.gif
---
width: 35em
name: ConvLNP_kernel_gp_text
alt: ConvLNP trained on GPs with RBF,Matern,periodic kernel
---
Similar to {numref}`ConvLNP_single_gp_extrap_text` but the training was performed on all data simultaneously.
```

{numref}`ConvLNP_kernel_gp_text` demonstrates that ConvLNP performs quite well in this harder setting.
Indeed, it seems to model process using the periodic kernel when the number of context points is small but quickly (around 10 context points) recovers the correct underlying kernel.
Note that in {numref}`ConvLNP_kernel_gp_text` we are plotting the posterior predictive for the sampled GP, rather than the actual, non-GP posterior predictive process.

[should we also add the results of {numref}`ConvLNP_vary_gp` to show that not great when large/ uncountable number of kernels?]

Again, we consider the performance of the ConvLNP in the image setting.
In {numref}`ConvLNP_img_text` we see that the ConvLNP does a reasonable job producing samples when the context sets are uniformly subsampled from images, but struggles with the "structured" context sets, e.g. when the left or bottom halves of the image are missing.
Moreover, the ConvLNP is able to produce samples in the generalisation setting (ZSMM), bit these are not always coherent, and include some strange artifacts that seem more similar to sampling the MNIST "texture" than coherent digits.

```{figure} ../gifs/ConvLNP_img.gif
---
width: 45em
name: ConvLNP_img_text
alt: ConvLNP on CelebA, MNIST, ZSMM
---
Samples from posterior predictive of an ConvCNP for CelebA $32\times32$, MNIST, ZSMM.
```

As discussed in  {ref}`the "CNPG" issue section <issues-cnpfs>`, members of the CNPF could not be used to generate coherent samples, nor model non-Gaussian posterior predictive distributions.
{numref}`ConvLNP_marginal_text` (right) demonstrates that, as expected, ConvLNP is able to produce non-Gaussian predictives for pixels, with interesting bi-modal and heavy-tailed behaviours.

```{figure} ../images/ConvLNP_marginal.png
---
width: 20em
name: ConvLNP_marginal_text
alt: Samples from ConvLNP on MNIST and posterior of different pixels
---
Samples form the posterior predictive of ConvCNPs on MNIST (left) and posterior predictive of some pixels (right).
```


[REVERT PLOTS OF CONVLNP, I made a small modificiation which looks bad...]

### Issues and Discussion

We have seen how members of the LNPF utilise a latent variable to define a predictive distribution, thus achieving structured and expressive predictive distributions over target sets.
Despite these advantages, the LNPF suffers from important drawbacks:

* Training procedures require several approximations, often resulting in sub-par models.
* Evaluating objective functions relies on sampling procedures, which often leads to high variance during training.
* They are more memory and computationally demanding, often requiring many samples to estimate desired quantities.
* Since our performance objectives are lower-bounds (or conservative estimates), it is often difficult to rigorously evaluate and compare different models.

Despite these challenges, the LNPF defines a useful and powerful class of models, that can be deployed in several important settings.
Thus, it is not always clear whether it would be better to deploy a member of the CNPF or LNPF.
For example, if samples are not required for a particular application, and we have reason to believe a parametric distribution may be a good description of the likelihood, it may well be that a CNPF would be preferable.
Conversely, LNPF members are particularly useful when sampling, thus being well-suited to applications such as Thompson sampling (e.g. for contextual bandits) and Bayesian optimisation.

Finally, we believe there is an exciting area of research in developing NPF members that enjoy the best of both worlds.
Some examples of potential avenues for investigation are:
1. Improving training procedures for the LNPF using ideas from importance sampling and unbiased estimators of the marginal likelihood.
2. Considering flow-based parameterisations that admit closed form likelihood computations, while potentially maintaining structure in the predictive distribution for the target set.

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.
