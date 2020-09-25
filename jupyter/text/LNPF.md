# Latent NPFs

## Overview

We concluded the previous section by noting two important drawbacks of the CNPF:

* The predictive distribution is factorised across target points, and thus can neither account for correlations in the predictive nor (as a result) produce "coherent" samples from the predictive distribution.
* The predictive distribution requires specification of a particular parametric form (e.g. Gaussian).

In this section we discuss an alternative parametrisation of $p_\theta( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}; \mathcal{C})$ that still satisfies our desiredata for NPs, and addresses both of these issues.
The main idea is to introduce a latent variable $\mathbf{z}$ into the definition of the predictive distribution.
This leads us to the second major branch of the NPF, which we refer to as the Latent Neural Process Sub-family, or LNPF for short.
A graphical representation of the LNPF  is given in {numref}`graph_model_LNPs_text`.

```{figure} ../images/graphical_model_LNPF.png
---
width: 20em
name: graph_model_LNPs_text
alt: graphical model LNP
---
Probabilistic graphical model for LNPs.
```

```{admonition} Disclaimer$\qquad$ (Latent) Neural Processes Family
---
class: caution, dropdown
---
In this tutorial, we refer use the adjective "Latent" to distinguish the Latent NPF from the Conditional NPF. 
We then use term "neural process" to refer to both conditional neural processes and latent neural processes.
In the literature, however, the term "neural process" is used to refer to "latent neural processes".
As a result the models that we will discuss and call LNP, AttnLNP, and ConvLNP are found in the literature under the abbreviations NP, AttnNP, ConvNP.
```

To specify this family of models, we must define a few components:

* An encoder: $p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)$, which provides a _distribution_ over the latent variable $\mathbf{z}$ having observed the context set $\mathcal{C}$. As with other NPF, the encoder needs to be permutation invariant to correctly treat $\mathcal{C}$ as a set. A typical example is to first have a deterministic representation $R$ and then use it to output the mean and (log) standard deviations of a Gaussian distribution over $\mathbf{z}$.
* A decoder: $p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right)$, which provides predictive distributions conditioned on $\mathbf{z}$ and a target location $\mathbf{x}_{\mathcal{T}}$.
The decoder will usually be the same as the CNPF, but using a sample of the latent representation $\mathbf{z}$ and marginalizing them, rather than a deterministic representation.

Putting altogether:

%Here too, a typical choice for the decoder is as a Gaussian distribution.However, as we discuss below, choosing the decoder to have a Gaussian form in this case is far less restrictive than with the CNPF. We can now express the LNPF predictive distribution as

```{math}
:label: latent_likelihood
\begin{align}
p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}; \mathcal{C})
&= \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation}  \\
&= \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\theta}(y^{(t)} |  x^{(t)}, \mathbf{z}) \, \mathrm{d}\mathbf{z}  & \text{Factorisation}\\
&= \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)  \prod_{t=1}^{T} \mathcal{N} \left( y^{(t)};  \mu^{(t)}, \sigma^{2(t)} \right) \mathrm{d}\mathbf{z} & \text{Gaussianity}
\end{align}
```

```{admonition} Advanced$\qquad$Marginalisation and Factorisation $\implies$ Consistency
---
class: hint, dropdown
---
We show that like the CNPF, members of the LNPF also specify consistent stochastic processes conditioned on a fixed context set $\mathcal{C}$. 

(Consistency under permutation) Let $\mathbf{x}_{\mathcal{T}} = \{ x^{(t)} \}_{t=1}^T$ be the target inputs and $\pi$ be any permutation of $\{1, ..., T\}$. Then the predictive density is:

$$
\begin{align}
    p_\theta(y^{(1)}, ..., y^{(T)} | x^{(1)}, ..., x^{(T)}; \mathcal{C}) &= \int p_{\theta} ( \mathbf{z}) \prod_{t=1}^{T} p_{\theta}(y^{(t)} |  x^{(t)}, \mathbf{z}) \, \mathrm{d}\mathbf{z} \\
    &= \int p_{\theta} ( \mathbf{z}) \prod_{t=1}^{T} p_{\theta}(y^{(\pi(t))} |  y^{(\pi(t))}, \mathbf{z}) \, \mathrm{d}\mathbf{z} \\
    &= p_\theta(y^{(\pi(1))}, ..., y^{(\pi(T))} | x^{(\pi(1))}, ..., x^{(\pi(T))}; \mathcal{C})
\end{align}
$$

since multiplication is commutative. 

(Consistency under marginalisation)  Consider two target inputs, $x^{(1)}, x^{(2)}$. Then by marginalising out the second target output, we get:

$$
\begin{align}
    \int p_\theta(y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}; \mathcal{C}) \, \mathrm{d}y^{(2)} &= \int \int p_\theta(y^{(1)}| x^{(1)}, \mathbf{z})p_\theta(y^{(2)}| x^{(2)}, \mathbf{z}) p_\theta(\mathbf{z} | \mathcal{C}) \, \mathrm{d}\mathbf{z} \mathrm{d}y^{(2)} \\
    &= \int  p_\theta(y^{(1)}| x^{(1)}, \mathbf{z}) p_\theta(\mathbf{z}| \mathcal{C}) \int p_\theta(y^{(2)}| x^{(2)}, \mathbf{z})  \, \mathrm{d}y^{(2)} \mathrm{d}\mathbf{z}  \\
    &= \int  p_\theta(y^{(1)}| x^{(1)}, \mathbf{z}) p_\theta(\mathbf{z}| \mathcal{C}) \mathrm{d}\mathbf{z}  \\
    &= p_\theta(y^{(1)}| x^{(1)}; \mathcal{C})
\end{align}
$$

which shows that the predictive distribution obtained by querying an LNPF member at $x^{(1)}$ is the same as that obtained by querying it at $x^{(1)}, x^{(2)}$ and then marginalising out the second target point. Of course, the same idea works with collections of any size, and marginalising any subset of the variables.
```

Now, you might be worried that we have still made both the factorisation and Gaussian assumptions!
However, while $p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right)$ is still factorised, the predictive distribution we are actually interested in, $p_\theta( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}; \mathcal{C})$, is no longer due to the marginalisation of $\mathbf{z}$, thus addressing the first problem we associated with the CNPF.
Moreover, the predictive distribution is no longer Gaussian either.
In fact, since the predictive now has the form of an _infinite mixture of Gaussians_, potentially _any_ predictive density can be represented (i.e. learned) by this form.
This is great news, as it (conceptually) relieves us of the burden of choosing / designing a bespoke likelihood function when deploying the NPF for a new application!

However, there is an important drawback. The key difficulty with the LNPF is that the likelihood we defined in Eq.{eq}`latent_likelihood` is no longer _analytically tractable_.
We now discuss how to train members of the LNPF in general.
After discussing several training procedures, we'll introduce extensions of each of the CNPF members discussed in the previous chapter to their corresponding member of the LNPF.

(training_lnpf)=
## Training LNPF members

Ideally, we would like to directly maximize the likelihood defined in Eq.{eq}`latent_likelihood` to optimise the parameters of the model.
However, the integral over $\mathbf{z}$ renders this quantity intractable, so we must consider alternatives.
In fact, this story is not new, and the same issues arise when considering other latent variable models, such as variational auto-encoders (VAEs).

The question of how best to train LNPF members is still open, and there is ongoing research in this area.
In this section, we will cover two methods for training LNPF models, but each have their flaws, and deciding which is the preferred training method must often be answered empirically.
Here is a brief summary / preview of both methods which are described in details in the following sections:

```{list-table} Summary of training methods for LNPF models
---
header-rows: 1
stub-columns: 1
name: summary_training
---

* - Training Method
  - Approximation of Log Likelihood
  - Biased
  - Variance
  - Empirical Performance
* - {ref}`NPML <NPML>` {cite}`foong2020convnp`
  - Sample Estimate
  - Yes
  - Large
  - Usually better
* - {ref}`NPVI <NPML>` {cite}`garnelo2018neural`
  - Variational Inference
  - Yes
  - Small
  - Usually worst
```


```{admonition} Disclaimer$\qquad$ Chronology
---
class: caution, dropdown
---
In this tutorial, the order in which we introduce the objective functions does not follow the chronology in which they were originally introduced in the literature.
We begin by describing an approximate maximum-likelihood procedure, which was recently introduced by {cite}`foong2020convnp` due to its simplicity, and its relation to the CNPF training procedure.
Following this, we introduce a variational inference-inspired approach, which was proposed earlier by {cite}`garnelo2018neural` to train members of the LNPF.
```

(NPML)=
### Neural Process Maximum Likelihood (NPML)


First, let's consider a direct approach to optimising the log-marginal predictive likelihood of LNPF members.
While this quantity is no longer tractable (as it was with members of the CNPF), we can derive an estimator using Monte-Carlo sampling:

```{math}
:label: npml
\begin{align}
\log p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}; \mathcal{C})
&= \log \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation} \\
& \approx \log \left( \frac{1}{L} \sum_{l=1}^{L} \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z}_l \right) \right) & \text{Monte-Carlo approximation} \\
& = \hat{\mathcal{L}}_{\mathrm{ML}}
\end{align}
```

where each $\mathbf{z}_l \sim p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)$. 

````{admonition} Implementation$\qquad$LogSumExp
---
class: dropdown
---
In practice, manipulating directly probabilities is prone to numerical instabilities, e.g., multiplying probabilities as in Eq.{eq}`npml` will often underflow. 
As a result one should manipulate probabilities in the log domain:

```{math}
\begin{align}
\log p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}; \mathcal{C})
&\approx \log \left( \frac{1}{L} \sum_{l=1}^{L} \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z}_l \right) \right) & \text{Monte-Carlo} \\
& = \log \left( \sum_{l=1}^{L} \exp \left(  \sum_{t=1}^{T} \log p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z}_l \right) \right) \right) - \log L & \text{LogSumExp trick}\\
& = \text{LogSumExp}_{l=1}^{L} \left( \sum_{t=1}^{T} \log p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z}_l \right) \right) - \log L,
\end{align}
```

where numerical stable implementations of [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations) can be found in most frameworks. 
````

Eq.{eq}`npml` provides a simple-to-compute objective function for training LNPF-members, which we can then use with standard optimisers to learn the model parameters $\theta$.
The final (numerically stable) pseudo-code for NPML is given in {numref}`npml_pseudocode`:

```{figure} ../images/alg_npml.png
---
width: 35em
name: npml_pseudocode
alt: Pseudo-code NPML.
---
Pseudo-code for a single training step of a LNPF member with NPML.
```

NPML is conceptually very simple as it directly approximates the training procedure of the CNPF, in the sense that it targets the same predictive likelihood during training.
Moreover, it tends to work well in practice, typically leading to models achieving good performance.
However, it suffers from two important drawbacks:
1. *Bias*: When applying the Monte-Carlo approximation, we have employed an unbiased estimator to the predictive likelihood. However, in practice we are interested in the _log_ likelihood. Unfortunately, the log of an unbiased estimator is not itself unbiased. As a result, NPML is a _biased_ (conservative) estimator of the true log-likelihood.
2. *High Variance*: In practice it turns out that NPML is quite sensitive to the number of samples $L$ used to approximate it. In both our GP and image experiments, we find that on the order of 20 samples are required to achieve "good" performance. Of course, the computational and memory costs of training scale linearly with $L$, often limiting the number of samples that can be used in practice.

Unfortunately, decreasing the number of samples $L$ needed to perform well turns out to be quite difficult, and is an open question in training latent variable models in general.
However, we next describe an alternative training procedure that typically works well with fewer samples.


### Neural Process Variational Inference (NPVI)

NPVI is a training procedure proposed by {cite}`garnelo2018neural`, which takes inspiration from the literature on variational inference (VI).
The central idea behind this objective function is to use _posterior sampling_ to reduce the variance of NPML.
For a better intuition regarding this, note that NPML is defined using an expectation against $p_{\theta}(\mathbf{z} | \mathcal{C})$.
The idea in posterior sampling is to use the whole task, including the target set, $\mathcal{D} = \mathcal{C} \cup \mathcal{T}$ to produce the distribution over $\mathbf{z}$, thus leading to more informative samples and lower variance objectives.


In our case, the posterior distribution from which we would like to sample is $p(\mathbf{z} | \mathcal{C}, \mathcal{T})$, i.e., the distribution of the latent variable having observed _both_ the context and target sets.
Unfortunately, this posterior is intractable.
To address this, {cite}`garnelo2018neural` propose to replace the true posterior with simply passing both the context and target sets through the encoder, i.e.

```{math}
---
label: approximate_posterior
---
\begin{align}
p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) \approx p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)
\end{align}
```


```{admonition} Note$\qquad p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) \neq p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)$
---
class: note, dropdown
---
Note that these two distributions are different.
We can compute $p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)$ by simply passing both $\mathcal{C}$ and $\mathcal{T}$ through the model encoder.
One the other hand, the posterior $p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right)$ is computed using Bayes' rule and our model definition:

$$
\begin{align}
p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) = \frac{p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right)}{ \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right) \mathrm{d} \mathbf{z}}.
\end{align}
$$

Recalling that our decoder is defined by a complicated, non-linear neural network, we can see that this posterior is intractable, as it involves an integration against complicated likelihoods.
```

We can now derive the final objective function, which is a _lower bound_ to the log marginal likelihood, by (i) introducing the approximate posterior as a sampling distribution, and (ii) employing a straightforward application of [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality).

```{math}
:label: npvi
\begin{align}
\log p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= \log \int p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)  p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Marginalisation} \\
& = \log \int p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \frac{p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)}{p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)} p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) \mathrm{d}\mathbf{z} & \text{Importance Weight} \\
& \geq \int p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \left( \log p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) + \log \frac{p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)}{p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)} \right) & \text{Jensen's inequality} \\
& = \mathbb{E}_{\mathbf{z} \sim p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)} \left[ \log p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) \right] - \mathrm{KL} \left( p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \| p_{ \theta} \left( \mathbf{z} | \mathcal{C} \right) \right) \\
& = \mathcal{L}_{\mathrm{VI}}
\end{align}
```

where $\mathrm{KL}(p \| q)$ is the Kullback-Liebler (KL) divergence between two distributions $p$ and $q$, and we have used the shorthand $p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) = \prod_{t=1}^{T} p_{\theta} \left( y^{(t)} | x^{(t)}, \mathbf{z} \right)$ to ease notation.
Let's consider what we have achieved in Eq. {eq}`npvi`.
% We have taken our original objective, and re-expressed it as an expectation with respect to $p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)$, which now has access to all of $\mathcal{D}$, as opposed to the NPML objective, where the expectation was only with respect to $p_{\theta} \left( \mathbf{z} | \mathcal{C} \right)$.


```{admonition} Test Time
---
class: attention
---
Of course, we can only sample from this approximate posterior during _training_, when we have access to both the context _and_ target sets.
At test time, we will only have access to the context set, and so the forward pass through the model will be equivalent to that of the model when trained with NPML, i.e., we will only pass the context set through the encoder.
This is an important detail of NPVI: forward passes at meta-train time look different than they do at meta-test time!
```


```{admonition} Advanced$\qquad$LNPF as Amortised VI
---
class: dropdown, hint
---
The above procedure can be seen as a form of _amortised_ VI.
Amortised VI is a method for performing approximate inference and learning in probabilistic latent variable models, where an _inference_ network is trained to approximate the true posterior distributions.

There are many resources available on (amortised) VI (e.g., [Jaan Altosaar's VAE tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/), [the Spectator's](http://blog.shakirm.com/2015/01/variational-inference-tricks-of-the-trade/) take, or this [review paper on modern VI](https://arxiv.org/pdf/1711.05597.pdf)), and we encourage readers unfamiliar with the concept to take the time to go through some of these.
For our purposes, the following intuitions should suffice:

Assume that we have a latent variable model with a prior $p(\mathbf{z})$, and a conditional likelihood for observations $y$, written $p_{\theta} \left(y | \mathbf{z} \right)$.
The central idea in amortised VI is to introduce an _inference network_, denoted $q_{\phi}(\mathbf{z} | y)$, which maps observations to distributions over $\mathbf{z}$.
We can then use $q_{\phi}$ to derive a lower bound to the log-marginal likelihood, just as we did above:

$$
\begin{align}
\log p_{\theta}(y)
&= \log \int p_{\theta} \left( \mathbf{z} , y \right)\mathrm{d}\mathbf{z} \\
& = \log \int q_{\phi} \left( \mathbf{z} | y \right) \frac{p_{\theta} \left( \mathbf{z} , y \right)}{q_{\phi} \left( \mathbf{z} | y \right)} \mathrm{d}\mathbf{z} \\
& \geq \mathbb{E}_{\mathbf{z} \sim q_{\phi} \left( \mathbf{z} | y \right)} \left[ \log p_{\theta} \left( y | \mathbf{z} \right) \right] - \mathrm{KL} \left( q_{\phi} \left( \mathbf{z} | y \right) \| p_{ \theta} \left( \mathbf{z} \right) \right).
\end{align}
$$

In the VI terminology, this lower bound is commonly known as the _evidence lower bound_ (ELBO).
So maximising the ELBO with respect to $\theta$ trains the model to optimise a lower bound on the log-marginal likelihood, which is a sensible thing to do.
Moreover, it turns out that maximising the ELBO with respect to $\phi$ minimises the KL divergence between $q_{\phi} \left( \mathbf{z} | y \right)$ and the true posterior $p_\theta(\mathbf{z} | y) = p_\theta(y | \mathbf{z}) p_\theta(\mathbf{z}) / p_\theta(y)$, so we can think of $q_{\phi}$ as approximating the true posterior in a meaningful way.

In the NPF, to approximate the desired posterior, we can introduce a network that maps datasets to distributions over the latent variable.
We already know how to define such networks in the NPF -- that's exactly what the encoder $p_{\theta}(\mathbf{z} | \mathcal{C})$ does!
In fact, NPVI proposes to use the encoder as the inference network when training LNPF members.

So we can view Eq.{eq}`npvi` as performing amortised VI for any member of the LNPF.
The twist on standard amortised VI is that here, we are sharing $q_{\phi}$ with a part of the model itself, since $p_{\theta}(\mathbf{z} | \mathcal{D})$ plays the dual role of being an approximate posterior $q_{\phi}$, and also defining the _conditional prior_ having observed $\mathcal{D}$. This somewhat complicates our understanding of the procedure, and could lead to unintended consequences.
```

When both the encoder and inference network parameterise Gaussian distributions over $\mathbf{z}$ (as is standard), the KL-term can be computed analytically.
Hence we can derive an unbiased estimator to Eq.{eq}`npvi` by taking samples from $p_{\theta} \left( \mathbf{z} | \mathcal{D} \right)$ to estimate the first term on the RHS.
{numref}`npvi_pseudocode` provides the pseudo-code for a single training iteration for a LNPF member, using NPVI as the target objective.

```{figure} ../images/alg_npvi.png
---
width: 35em
name: npvi_pseudocode
alt: Pseudo-code NPVI.
---
Pseudo-code for a single training step of a LNPF member with NPVI.
```

```{admonition} Warning$\qquad$Biased estimators
---
class: warning, dropdown
---
Importantly, it is not the case that NPVI avoids the "biased estimator" issue of NPML.
Rather, NPML defines a biased (conservative) estimator of the desired quantity, and NPVI produces an unbiased estimator of a strict lower-bound of the same quantity.
In both cases, unfortunately, we do not have unbiased estimators of the quantity we really care about --- $\log p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$.

Another consequence of this is that it is challenging to evaluate the models quantitatively, since our performance metrics are only ever lower bounds to the true model performance, and quantifying how tight those bounds are is quite challenging.
```


To better understand the NPVI objective, it is important to note --- see the box below --- that it can be rewritten as the difference between the desired log marginal likelihood and a KL divergence between approximate and true posterior:

```{math}
:label: npvi2
\mathcal{L}_{VI} = \log p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C}) - \mathrm{KL} \left( p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \| p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) \right) 
```

The NPVI can thus be seen as maximizing the desired log marginal likelihood as well as forcing the approximate and true posterior to be similar.


```{admonition} Advanced$\qquad$Relationship between NPML and NPVI
---
class: hint, dropdown
---
There is a close relationship between the NPML and NPVI objectives.
To see this, we denote the _normalising constant_ for the distribution $p_{\theta} \left( \mathbf{y}_{\mathcal{T}}, \mathbf{z} | \mathbf{x}_{\mathcal{T}}, \mathcal{C} \right)$ as

$$
\begin{align}
  Z = \int p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) \mathrm{d}\mathbf{z}.
\end{align}
$$

Now, we can rewrite the NPVI objective as

$$
\begin{align}
\mathcal{L}_{VI} & (\theta ; D) = \mathbb{E}_{p_{\theta}(\mathbf{z} | \mathcal{D})} \left[ \log p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) \right] - \mathrm{KL} \left( p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \| p_{ \theta} \left( \mathbf{z} | \mathcal{C} \right) \right) \\
& = \mathbb{E}_{p_{\theta}(\mathbf{z} | \mathcal{D})} \left[ \log p_{\theta} \left( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathbf{z} \right) + \log p_{\theta} \left( \mathbf{z} | \mathcal{C} \right) - \log p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \right] \\
& = \mathbb{E}_{p_{\theta}(\mathbf{z} | \mathcal{D})} \left[ \log Z + \log \frac{1}{Z} p_{\theta} \left( \mathbf{y}_{\mathcal{T}}, \mathbf{z} | \mathbf{x}_{\mathcal{T}}, \mathcal{C} \right) - \log p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \right] \\
& = \log p_{\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C}) - \mathrm{KL} \left( p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \| \frac{1}{Z} p_{\theta} \left( \mathbf{y}_{\mathcal{T}}, \mathbf{z} | \mathbf{x}_{\mathcal{T}}, \mathcal{C} \right) \right) \\
& = \mathcal{L}_{ML}(\theta; \mathcal{D}) - \mathrm{KL} \left( p_{\theta} \left( \mathbf{z} | \mathcal{D} \right) \| p \left( \mathbf{z} | \mathcal{C}, \mathcal{T} \right) \right) .
\end{align}
$$

Thus, we can see that $\mathcal{L}_{VI}$ is equal to $\mathcal{L}_{ML}$ (with infinitely many samples) up to an additional KL term.
This KL term has a nice interpretation as encouraging consistency among predictions with different context sets, which is the kind of consistency _not_ baked into the NPF.

However, this term can also be a _distractor_.
When dealing with NPF members, we are typically not interested in the latent variable $\mathbf{z}$, and most considered tasks require only a "good" approximation to the predictive distribution over $\mathbf{y}_{\mathcal{T}}$.
Therefore, given only finite model capacity, and finite data, it may be preferable to focus all the model capacity on achieving the best possible predictive distribution (which is what $\mathcal{L}_{ML}$ focuses on), rather than focusing on the distribution over $\mathbf{z}$, as encouraged by the KL term introduced by $\mathcal{L}_{VI}$.

Notice that if our approximate posterior can recover the true posterior, then the KL term is zero, and we are exactly optimising the log-marginal likelihood.
In practice, however, the inequality holds, meaning that we are only optimising a lower-bound to the quantity we actually care about. Moreover, it is often quite difficult to know how tight this bound may be.
```


As we have discussed, the most appealing property of NPVI is that it utilises _posterior sampling_ to reduce the variance of the Monte-Carlo estimator of the intractable expectation.
This means that often we can get away with training models taking just a single sample, resulting in computationally and memory efficient training procedures.
However, it also comes with several drawbacks, which can be roughly summarised as follows:

* NPVI is focused on approximating the posterior distribution (see Eq.{eq}`npvi2` ). However, in the NPF setting, we are typically only interested in the predictive distribution $p(\mathbf{y}_T | \mathbf{x}_T, \mathcal{C})$, and it is unclear whether focusing our efforts on $\mathbf{z}$ is beneficial to achieving higher quality predictive distributions.
* In NPVI, the encoder plays a dual role: it is both part of the model, and used for posterior sampling. This fact introduces additional complexities in the training procedure, and it may be that using the encoder as an approximate posterior has a detrimental effect on the resulting predictive distributions.



As we shall see below, it is often the case that models trained with NPML produce better fits than equivalent models trained with NPVI, at the cost of additional computational and memory costs of training.
Moreover, using NPVI often requires additional tricks and contraints to get good descent performance.


Armed with procedures for training LNPF-members, we turn our attention to the models themselves.
In particular, we next introduce the latent-variable variant of each of the conditional models introduced in the previous section, and we shall see that having addressed the training procedures, the extension to latent variables is quite straightforward from a practical perspective.



## Latent Neural Process (LNP)

```{figure} ../images/computational_graph_LNPs.png
---
width: 25em
name: computational_graph_LNPs_text
alt: Computational graph LNP
---
Computational graph for LNPS.
```

The latent neural process {cite}`garnelo2018neural` is the latent counterpart of the CNP, and the first member of the LNPF proposed in the literature.
Given the vector $R$, which is computed as in the CNP, we simply pass it through an additional MLP to predict the mean and variance of the latent representation $\mathbf{z}$, from which we can produce samples.
The decoder then has the same architecture as that of the CNP, and we can simply pass samples of $\mathbf{z}$, together with desired target locations, to produce our predictive distributions.
{numref}`computational_graph_LNPs_text` illustrates the computational graph of the LNP.

```{admonition} Implementation$\qquad$Parameterising the Observation Noise
---
class: note, dropdown
---
An important detail in the parameterisation of LNPF members is how the predictive standard deviation is handled, there are 2 possibilities
- *Heteroskedastic noise*: As with CNPF members, the decoder parameterises a mean $\mu^{(t)}$ and a standard deviation $\sigma^{(t)}$ for each target location $x^{(t)}$.
- *Homoskedastic noise*: The decoder only parameterises a mean $\mu^{(t)}$, while the standard deviation $\sigma$ is a global (learned) observation noise parameter, i.e., it is shared for all target locations.
Note that the uncertainty of the posterior predictive at two different target locations can still be very different.
Indeed, it is influenced by the variance in the *means* $\mu^{(t)}$, which arises from the uncertainty in the latent variable.

In practice, a heteroskedastic noise usually performs better than the homoskedastic variant.
%, but additional care has to be taken when implementing such models.Namely, it requires to use a lower bound for the predicted standard deviation.To illustrate why this is the case, imagine training a LNP with heteroskedastic noise on MNIST. In the MNIST dataset, some pixels close to the borders are always black, as a result if a model can predict one such pixel as black with no uncertainty ($\sigma^{(t)}=0$) it will get in

```

````{admonition} Implementation$\qquad$"Lower Bounding" Standard Deviations
---
class: dropdown, note
---

In the LNPF literature, it is common to "lower bound" the standard deviations of distributions output by models.
This is often achieved by parameterising the standard deviation as

$$
\begin{align}
	\sigma = \epsilon + (1 - \epsilon) \ln (1 + \exp (f_\sigma)),
\end{align}
$$

where $\epsilon$ is a small real number (e.g. 0.001), and $f_\sigma$ is the log standard deviation output by the model.
This "lower bounding" is often used in practice for the standard deviations of both the latent and predictive distributions.
In fact Le et al.{cite}`le2018empirical` find that this consistently improves performance of the AttnLNP, and recommend it as best practice.
Following the literature, the results displayed in this tutorial all employ such lower bounds.
However, we note that, in our opinion, there is no conceptual justification for this trick, and it is indicative of flaws in the models and or training procedures.

````

Throughout the section we train LNPs with NPVI as in the original paper.
Below, we show the predictive distribution of an LNP trained on samples from the RBF-kernel GP, as the number of observed context points from the underlying function increases.


```{figure} ../gifs/LNP_rbf.gif
---
width: 35em
name: LNP_rbf_text
alt: LNP on GP with RBF kernel
---
Samples from posterior predictive of LNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`graph_model_LNPs_text` shows that the latent variable indeed enables coherent sampling from the posterior predictive.
In fact, within the range $[-1, 1]$ the model produces very nice samples, that seem to properly mimic those from the underlying process.
Nevertheless, here too we see that the LNP suffers from the same underfitting issue as discussed with CNPs.
We again see the tendency to overestimate the uncertainty, and often not pass through all the context points with the mean functions.
Moreover, we can observe that beyond the $[-1, 1]$, the model seems to "give up" on the context points and uncertainty, despite having been trained on the range $[-2, 2]$.

````{admonition} Note$\qquad$NPVI vs NPML
---
class: dropdown, note
--- 
In the main text we trained LNP with NPVI as in the original paper, we compare the results with NPML on different kernels below.

```{figure} ../gifs/singlegp_LNP_LatLBTrue_SigLBTrue.gif
---
width: 35em
name: LNP_all_both_objectives_text
alt: LNP on all kernels with both objectives
---
Predictive distributions of LNPs (Blue) and the oracle GP (Green) with (top) RBF, (center) periodic, and (bottom) Noisy Matern kernels. Models were trained with (left) NPML and (right) NPVI.
```

{numref}`LNP_all_both_objectives_text` illustrates several interesting points that we tend to see in our experiments.
First, as with the CNP, the model tends to underfit, and in particular fails catastrophically with the periodic kernel.
In some cases, e.g., when trained on the RBF kernel with NPVI, the model seems to collapse the uncertainty arising from the latent variable in certain regions, and rely entirely on the observation noise.
In our experiments, we observe that this tends to occur when training with NPVI, but not with NPML.
Finally, we can see that NPML tends to produce predictive distributions that fit the data better, and tend to have lower uncertainty near the context set points.
````

Let us now consider the image experiments as we did with the CNP.

```{figure} ../gifs/LNP_img_interp.gif
---
width: 30em
name: LNP_img_interp_text
alt: LNP on CelebA and MNIST
---
samples (means conditioned on different samples from the latent) of the posterior predictive of a LNP on CelebA $32\times32$ and MNIST.
The last row shows the standard deviation of the posterior predictive corresponding to the last sample.
```

From {numref}`LNP_img_interp_text` we see again that the latent variable enables relatively coherent sampling from the posterior predictive.
As with the CNP, the LNP still underfits on images as is best illustrated when the context set is half the image.


```{admonition} Details
---
class: note
---
Model details, training and more plots in {doc}`LNP Notebook <../reproducibility/LNP>`.
 We also provide pretrained models to play around with.
```


## Attentive Latent Neural Process (AttnLNP)


The Attentive LNPs {cite}`kim2019attentive` is the latent counterpart of AttnCNPs.
Differently from the LNP, the AttnLNP _added_ a "latent path" in addition to (rather than instead of) the deterministic path.
The latent path is implemented with the same method as LNPs, i.e. a mean aggregation followed by a parametrization of a Gaussian.
In other words, even though the deterministic representation $R^{(t)}$ is target specific, the latent representation $\mathbf{z}$  is target independent as seen in the computational graph ({numref}`computational_graph_AttnLNPs_text`).

```{figure} ../images/computational_graph_AttnLNPs.png
---
width: 25em
name: computational_graph_AttnLNPs_text
alt: Computational graph AttnLNP
---
Computational graph for AttnLNPS. 
```

Throughout the section we train AttnLNPs with NPVI as in the original paper.
Below, we show the predictive distribution of an AttnLNP trained on samples from RBF, periodic, and noisy Matern kernel GPs, again viewing the predictive as the number of observed context points is increased.

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
* *Kinks*: The samples do not seem to be smooth, and we see "kinks" that are similar (though even more pronounced) than in {numref}`AttnCNP_single_gp_text`.
* *Collapse to AttnCNP*: In many places the AttnLNP seems to collpase the distribution around the latent variable, and express all its uncertainty via the observation noise. This tends to occur more often for the AttnLNP when trained with NPVI rather than NPML.

````{admonition} Note$\qquad$NPVI vs NPML
---
class: dropdown, note
---
As with the LNP, we compare the performance of the AttnLNP when trained with NPVI and NPML.

```{figure} ../gifs/singlegp_AttnLNP_LatLBTrue_SigLBTrue.gif
---
width: 35em
name: AttnLNP_all_both_objectives_text
alt: AttnLNP on all kernels with both objectives
---
Predictive distributions of AttnLNPs (Blue) and the oracle GP (Green) with (top) RBF, (center) periodic, and (bottom) Noisy Matern kernels. Models were trained with (left) NPML and (right) NPVI.
```

Here we see that the AttnLNP tends to "collapse" for both the RBF and noisy Matern kernels when trained with NPVI.
In contrast, when trained with NPML it tends to avoid this behaviour.
This is consistent with what we observe more generally in our experiments.
````


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
class: note
---
Model details, training and more plots in {doc}`AttnLNP Notebook <../reproducibility/AttnLNP>`.
 We also provide pretrained models to play around with.
```


## Convolutional Latent Neural Process (ConvLNP)

The Convolutional LNP {cite}`foong2020convnp` is the latent counterpart of the ConvCNP.
In contrast with the AttnLNP, the latent path _replaces_ the deterministic one (as with LNP), resulting in a latent functional representation (a latent stochastic process) instead of a latent vector valued variable.

Another way of viewing the ConvLNP, which is useful in gaining an intuitive understanding of the computational graph (see {numref}`computational_graph_ConvLNPs_CpnvCNPs_text`) is as consisting of two stacked ConvCNPs: the first takes in context sets and outputs a latent stochastic process.
The second takes as input a sample from the latent process and models the posterior predictive conditioned on that sample.

```{figure} ../images/computational_graph_ConvLNPs_ConvCNPs.png
---
width: 25em
name: computational_graph_ConvLNPs_CpnvCNPs_text
alt: Computational graph ConvLNP using two ConvCNPs
---
Computational graph for ConvLNPs. 
```

````{admonition} Implementation$\qquad$ConvLNP without ConvCNP
---
class: note, dropdown
---
In our implementation we do not use two ConvCNPs to implement the ConvLNP.
Instead, we use the same computational graph than for ConvCNPs but split the CNN into two. 
The output of the first CNN is a mean and a (log) standard deviation at every position of the (discrete) signal, which are then used to parametrize independent Gaussian disitrubtions each positions.
The second CNN gets as input a sample from the first one, i.e., a discrete signal.

The computational graph for this implementation is shown in {numref}`computational_graph_ConvLNPs_text`.
Importantly both are mathematically equivalent, but using two ConvCNPs can be easier to understand and more modular, while using two CNNs avoids unecessary computations.

```{figure} ../images/computational_graph_ConvLNPs.png
---
width: 25em
name: computational_graph_ConvLNPs_text
alt: Computational graph ConvLNP
---
Computational graph for ConvLNPs without using ConvCNPs. 
```

````

One difficulty arises in training the ConvLNP with the NPVI objective, as it requires evaluating the KL divergence between two stochastic processes, which is a tricky proposition.
{cite}`foong2020convnp` propose a simple approach, that approximates this quantity by instead summing the KL divergences at each discretisation location.
However, as they note, the ConvLNP performs significantly better in most cases when trained with NPML rather than NPVI.
Throughout this section we will thus use NPML instead of NPVI.

```{admonition} Note$\qquad$Global Latent Representations
---
class: note, dropdown
---
In this tutorial, we consider a simple extension to the ConvLNP proposed by Foong et. al {cite}`foong2020convnp`, which includes a _global latent representation_ as well.
The global representation is computed by average-pooling half of the channels in the latent function, resulting in a translation _invariant_ latent representation, in addition to the translation equivariant one.
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
Importantly, the data generating process is a mixture of GPs, and the true posterior predictive process (achieved by marginalising over the different kernels) is non-Gaussian.
<!-- Theoretically, this could still be modelled by a LNPF as the latent variables could represent the both the current kernel and its functions, but adding the global representation corresponds (intuitively) to allowing the ConvLNP to "switch" between kernels. -->


```{figure} ../gifs/ConvLNP_kernel_gp.gif
---
width: 35em
name: ConvLNP_kernel_gp_text
alt: ConvLNP trained on GPs with RBF,Matern,periodic kernel
---
Similar to {numref}`ConvLNP_single_gp_extrap_text` but the training was performed on all data simultaneously.
```

{numref}`ConvLNP_kernel_gp_text` demonstrates that ConvLNP performs quite well in this harder setting.
Indeed, it seems to model the predictive process using the periodic kernel when the number of context points is small but quickly (around 10 context points) recovers the correct underlying kernel.
Note how in the middle plot, the ConvLNP becomes progressively more and more "confident" that the process is periodic as more data is observed.
Note that in {numref}`ConvLNP_kernel_gp_text` we are plotting the posterior predictive for the sampled GP, rather than the actual, non-GP posterior predictive process.

Again, we consider the performance of the ConvLNP in the image setting.
In {numref}`ConvLNP_img_text` we see that the ConvLNP does a reasonable job producing samples when the context sets are uniformly subsampled from images, but struggles with the "structured" context sets, e.g. when the left or bottom halves of the image are missing.
Moreover, the ConvLNP is able to produce samples in the generalisation setting (ZSMM), but these are not always coherent, and include some strange artifacts that seem more similar to sampling the MNIST "texture" than coherent digits.

```{figure} ../gifs/ConvLNP_img.gif
---
width: 45em
name: ConvLNP_img_text
alt: ConvLNP on CelebA, MNIST, ZSMM
---
Samples from posterior predictive of an ConvCNP for CelebA $32\times32$, MNIST, ZSMM.
```

As discussed in  {ref}`the 'Issues with the CNPF' section <issues-cnpfs>`, members of the CNPF could not be used to generate coherent samples, nor model non-Gaussian posterior predictive distributions.
In contrast, {numref}`ConvLNP_marginal_text` (right) demonstrates that, as expected, ConvLNP is able to produce non-Gaussian predictives for pixels, with interesting bi-modal and heavy-tailed behaviours.

```{figure} ../images/ConvLNP_marginal.png
---
width: 20em
name: ConvLNP_marginal_text
alt: Samples from ConvLNP on MNIST and posterior of different pixels
---
Samples form the posterior predictive of ConvCNPs on MNIST (left) and posterior predictive of some pixels (right).
```

From {numref}`AttnLNP_img_text` we see that AttnLNP generates quite impressive samples, and exhibits descent sampling and good performances when the model does not require generalisation (CelebA $32\times32$, MNIST).
However, as expected, the model "breaks" for ZSMM as it still cannot extrapolate.

```{admonition} Details
---
class: note
---
Model details, training and more plots in {doc}`AttnLNP Notebook <../reproducibility/AttnLNP>`.
 We also provide pretrained models to play around with.
```


## Issues and Discussion

We have seen how members of the LNPF utilise a latent variable to define a predictive distribution, thus achieving structured and expressive predictive distributions over target sets.
Despite these advantages, the LNPF suffers from important drawbacks:

* The training procedure only optimises a biased objective or a lower bound to the true objective.
* Approximating the objective function requires sampling, which can lead to high variance during training.
* They are more memory and computationally demanding, requiring many samples to estimate the objective for NPML.
* It is difficult to quantitatively evaluate and compare different models, since only lower bounds to the log predictive likelihood can be estimated.

Despite these challenges, the LNPF defines a useful and powerful class of models.
Whether to deploy a member of the CNPF or LNPF depends on the task at hand.
For example, if samples are not required for a particular application, and we have reason to believe a parametric distribution may be a good description of the likelihood, it may well be that a CNPF would be preferable.
Conversely, it is crucial to use the LNPF if sampling or dependencies in the predictive are required, for example in Thompson sampling.

<!-- Finally, we believe there is an exciting area of research in developing NPF members that enjoy the best of both worlds.
Some examples of potential avenues for investigation are:
1. Improving training procedures for the LNPF using ideas from importance sampling and unbiased estimators of the marginal likelihood.
2. Considering flow-based parameterisations that admit closed form likelihood computations, while potentially maintaining structure in the predictive distribution for the target set. -->

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.
