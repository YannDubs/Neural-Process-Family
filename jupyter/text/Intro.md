# The Neural Process Family

Deep learning has revolutionised the world of data-driven prediction, but there are still plenty of problems where it isn't easily applicable. One such setting is the _small data_ regime where _good uncertainty estimation_ is required. Think, for example, of a doctor trying to predict a patient's treatment outcome. We might have some measurements of our patient's biophysical data since they were admitted to the hospital. However, just one patient's data isn't going to be enough to train a deep neural network. Furthermore, if the doctor is going to make a potentially life-changing treatment decision based on the network's predictions, it is crucial that the network knows how certain it is, instead of being confidently wrong --- something deep neural networks are prone to.

The **Neural Process Family** (NPF) is a collection of models (called neural processes (NPs)) that tackles both of these issues, by _meta-learning_ a distribution over predictors, also known as a _stochastic process_. Meta-learning allows neural processes to incorporate data from many related tasks (e.g. many different patients in our medical example) and the stochastic process framework allows the NPF to effectively represent uncertainty.

We will unpack both the terms "meta-learning" and "stochastic process" in the following section.
But before diving in, let's consider some tasks that the NPF is particularly well-suited for.

* **Predicting times-series data with uncertainty.**
Let's consider the task of restoring corrupted audio signals.
We are given a dataset $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$, where $x$ are the inputs (time) and $y$ are the outputs (sound wave amplitude), and our goal is to reconstruct the signal conditioned on $\mathcal{D}$.
If $\mathcal{D}$ is very sparse, there could be many reasonable reconstructions --- hence we should be wary of simply providing a single prediction, and instead include measures of uncertainty.
{numref}`ConvLNP` shows a neural process being used to sample plausible interpolations of simple time-series, both periodic and non-periodic.


```{figure} ../gifs/ConvLNP_norbf_gp_extrap.gif
---
width: 35em
name: ConvLNP
alt: Samples from ConvLNP trained on GPs
---
Sample functions from the predictive distribution of ConvLNPs (blue) and the oracle GP (green) with periodic and noisy Matern kernels.
```

* **Interpolating image data with uncertainty.** Imagine that we are given a satellite image of a region obscured by cloud-cover.
We might need to make predictions regarding what is "behind" the occlusions. For example, the [UNHCR](https://en.wikipedia.org/wiki/United_Nations_High_Commissioner_for_Refugees) might need to count the number of tents in a refugee camp to know how much food and healthcare supplies to send there.
If clouds are obscuring a large part of the image, we might be interested not just in a single interpolation, but in the entire probability distribution over plausible interpolations.
NPs can do exactly that.
{numref}`ConvCNP_superes_intro` shows a NP upscaling the resolution of an image by treating pixel locations that are "in between" the pixels of the input images as occluded pixels, i.e., by interpolating between pixels. The NP also provides uncertainty estimates, which aren't shown here.

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_intro
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```


```{admonition} Note$\qquad$Outline
---
class: note, dropdown
---
Our approach in this tutorial will be to walk through several prominent members of the NPF, highlighting their key advantages and drawbacks.
By accompanying the exposition with code and examples, we hope to (i) make the design choices and tradeoffs associated with each model clear and intuitive, and (ii) demonstrate both the simplicity of the framework and its broad applicability.

**This tutorial is split into three sections**.
This first page will give a broad, bird's eye view of the entire Neural Process Family.
We'll see that the family splits naturally into two sub-families: the {doc}`Conditional NPF <CNPF>`, and the {doc}`Latent NPF <LNPF>`.
These are covered in the following two pages, and there we'll get into more detail about the different architectures in each family.
In the Reproducibility section we provide the code to run all the models and generate all the plots, while the [github repository](https://github.com/YannDubs/Neural-Process-Family) contains the framework for implementing NPs.

Throughout the tutorial, we make liberal use of dropdown boxes to avoid breaking up the exposition.
Feel free to skip or skim those on a first reading.
```

## Meta Learning Stochastic Processes


```{figure} ../images/MetaStochasticProcess.png
---
width: 100%
name: meta_learning_sp
alt: Neural Processes as meta learning stochastic processes
---
Comparison between meta learning vs supervised learning, and modeling functions vs modeling stochastic processes. Neural Processes are in the lower-right quadrant.
```

### Meta-learning

In (deep) supervised learning, a neural network is trained to model a target function given some observations.
Specifically, a network is trained on a single dataset $\mathcal{C} := \{(x^{(c)}, y^{(c)})\}_{c=1}^C$ (which we will refer to as a **context set**). The trained network is then used as a predictor, $f(x)$.
A supervised learning algorithm can thus be seen as a map that takes datasets to a predictors $\mathcal{C} \mapsto f(x)$.
At test time, a prediction at a target location $x^{(t)}$ can be made by feeding it into the predictor to obtain $f(x^{(t)})$.
By doing so for all the test inputs (which we call **target inputs**) $\mathbf{x}_{\mathcal{T}} := \{x^{(t)}\}_{t=1}^T$, we get a set of predictions $f(\mathbf{x}_{\mathcal{T}}):= \{f(x^{(t)})\}_{t=1}^T$.
The predictor is evaluated by comparing $f(\mathbf{x}_{\mathcal{T}})$ to the ground truth **target outputs** $\mathbf{y}_{\mathcal{T}} := \{y^{(t)}\}_{t=1}^T$.
We will refer to a context and target set together as a *task*  $\mathcal{D} := (\mathcal{C}, \mathbf{x}_{\mathcal{T}}, \mathbf{y}_{\mathcal{T}})$.
This standard supervised learning process is visualised in the upper left quadrant of {numref}`meta_learning_sp`.

The idea of **meta-learning** is _learning to learn_, i.e., learning how to rapidly adapt to new supervised tasks.
The key insight is that, as we just saw, a supervised learning algorithm is itself a function from $\mathcal{C} \mapsto f(x)$.
As a result we can use supervised learning to model this function, hence the name *meta*.

To train a meta-learner, we need a large collection $\mathcal{M}= \{ \mathcal{D_i} \}_{i=1}^{N_{\mathrm{tasks}}}$ of related datasets --- a meta-dataset.
The result of meta-training on this meta-dataset is a supervised learning algorithm, i.e., a map $\mathcal{C} \mapsto f(x; \mathcal{C})$.
At meta-test time, we'll adapt the predictor to a task it has never seen before by providing it a new context set $\mathcal{C}$.
In this blog we will only consider cases where the map $\mathcal{C} \mapsto f(x; \mathcal{C})$ is parameterised by a neural network, meaning that the adaptation to a new task is done with a single forward pass, without any gradient updates!
The resulting predictor, $f(x; \mathcal{C})$, uses the information obtained during meta-learning to make predictions on this new task.
The whole meta-learning process is illustrated in the bottom left quadrant of {numref}`meta_learning_sp`.


Because it can share information across tasks, meta-learning is especially well-suited to situations where each task is a _small_ dataset, as in, e.g., few-shot learning.
However, if the context set is small, should we really expect to obtain a unique predictor, $f(x; \mathcal{C})$, from it?
To relate this back to our examples, if we only observe an audio signal at a few timestamps, or an image at a few pixels, can we really uniquely reconstruct the original?
What we need is to express our _uncertainty_, and this leads us naturally to _stochastic processes_.

```{admonition} Note$\qquad$Summary of Terminology
---
class: note, dropdown
---

```{list-table} Summary of NPF meta-learning terminology and notation
---
header-rows: 0
stub-columns: 1
name: notation
---

* - $\mathcal{C} := \{(x^{(c)}, y^{(c)})\}_{c=1}^C$
  - Context set
* - $\mathbf{x}_{\mathcal{T}} := \{x^{(t)}\}_{t=1}^T$
  - Target inputs
* - $\mathbf{y}_{\mathcal{T}} := \{y^{(t)}\}_{t=1}^T$
  - Target outputs
* - $\mathcal{T} := (\mathbf{x}_{\mathcal{T}}, \mathbf{y}_{\mathcal{T}}) = \{(x^{(t)}, y^{(t)})\}_{t=1}^T$
  - Target set
* - $f(\mathbf{x}_{\mathcal{T}}; \mathcal{C}) :=  \{f(x^{(t)}; \mathcal{C})\}_{t=1}^T$
  - Predictions
* - $\mathcal{D} := (\mathcal{C}, \mathcal{T}) = \{(x^{(n)}, y^{(n)})\}_{n=1}^{T+C}$
  - Dataset/task
* - $\mathcal{M} := \{ \mathcal{D}^{(i)} \}_{i=1}^{N_{\mathrm{tasks}}}$
  - Meta-dataset
```

(stochastic_processes)=
### Stochastic Process Prediction

We've seen that we can think of meta-learning as learning a map directly from context sets $\mathcal{C}$ to predictor functions $f(x; \mathcal{C})$.
However, there are many situations where a single predictor without error-bars isn't good enough.
Quantifying uncertainty is crucial for decision-making, and has many applications such as in model-based reinforcement learning, Bayesian optimisation and out-of-distribution detection.

Given target inputs $\mathbf{x}_{\mathcal{T}}$, what we need is not a single prediction $f(\mathbf{x}_{\mathcal{T}}; \mathcal{C})$, but rather a _distribution over predictions_ $p(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}; \mathcal{C})$.
As long as these distributions are consistent with each other for different choices of $\mathbf{x}_{\mathcal{T}}$, this is actually equivalent to specifying a distribution over _functions_, $f(x; \mathcal{C})$.
In mathematics, this is known as a **stochastic process** (SP).
Each predictor sampled from this distribution would represent a plausible interpolation of the data, and the diversity of the samples would reflect the _uncertainty_ in our predictions --- think back to {numref}`ConvLNP`.
Hence, **the NPF can be viewed as using neural networks to meta-learn a map from datasets to predictive stochastic processes**.
This is where the name Neural Process comes from, and is illustrated in the bottom right quadrant of {numref}`meta_learning_sp`.

```{admonition} Advanced$\qquad$Stochastic Process Consistency
---
class: hint, dropdown
---
In the previous discussion, we considered specifying a stochastic process (SP) by specifying $p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})$ for all finite collections of target inputs $\mathbf{x}_{\mathcal{T}}$.
Each  distribution for a given $\mathbf{x}_{\mathcal{T}}$ is referred to as a _finite marginal_ of the SP.
Can we stitch together all of these finite marginals to obtain a single SP?
The _Kolmogorov extension theorem_ tells us that we can, as long as the marginals are consistent with each other under _permutation_ and _marginalisation_:

To illustrate these consistency conditions, let's look at some artificial examples of finite marginals that are _not_ consistent.
Let $x^{(1)}, x^{(2)}$ be two target inputs, with $y^{(1)}, y^{(2)}$ the corresponding random outputs.

1. Consider a collection of finite marginals with $y^{(1)} \sim \mathcal{N}(0, 1)$ and $[y^{(1)}, y^{(2)}] \sim \mathcal{N}([10, 0], \mathbf{I})$. What is the mean of $y^{(1)}$?
2. Consider a collection with $[y^{(1)}, y^{(2)}] \sim \mathcal{N}([0, 0], \mathbf{I})$ and $[y^{(2)}, y^{(1)}] \sim \mathcal{N}([1, 1], \mathbf{I})$. What is the mean of $y^{(1)}$? What is the mean of $y^{(2)}$?

We see the problem here: inconsistent marginals lead to self-contradictory predictions!
In the first example, the marginals were not _consistent under marginalisation_: marginalising out $y^{(2)}$ from the distribution of $[y^{(1)}, y^{(2)}]$ did not yield the distribution of $y^{(1)}$.
In the second case, the marginals were not _consistent under permutation_: the distributions differed depending on whether you considered $y^{(1)}$ first or $y^{(2)}$ first.
Later we'll prove that this problem will never occur for NPs --- for a fixed context set $\mathcal{C}$, the NPF predictive distributions $p(y|x;\mathcal{C})$ always define a consistent stochastic process.

So far, we've only considered what happens when you fix the context set and vary the target inputs.
There is one more kind of consistency that we might expect SP predictions to satisfy: consistency among predictions with _different context sets_.
Consider two input-output pairs, $(x^{(1)}, y^{(1)})$ and $(x^{(2)}, y^{(2)})$.
The product rule of probability tells us that the joint predictive density must satisfy

$$
\begin{align}
p(y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}) &= p(y^{(1)}| x^{(1)}) p(y^{(2)}| x^{(2)}, y^{(1)}, x^{(1)}) \\
&= p(y^{(2)}| x^{(2)}) p(y^{(1)}| x^{(1)}, y^{(2)}, x^{(2)}).
\end{align}
$$

This essentially states that the distribution over $y^{(1)}, y^{(2)}$ obtained by autoregressive sampling should be independent of the order in which the sampling is performed.
Unfortunately, this is _not_ guaranteed to be the case NPs, i.e., it is possible that

$$
p(y^{(1)}| x^{(1)}) p(y^{(2)}| x^{(2)} ; \{x^{(1)},  y^{(1)} \} ) \neq p(y^{(2)}| x^{(2)}) p(y^{(1)}| x^{(1)} ; \{x^{(2)},  y^{(2)}\} ).
$$

The lack of consistency for different context sets is the reason we use the notation $p(y|x;\mathcal{C})$ instead of $p(y|x,\mathcal{C})$ as standard rules of probabilities cannot be used for $\mathcal{C}$.
In practice, NPs yields good predictive performance even though they can violate this consistency.
```

This point of view of NPs as outputting predictive stochastic processes is helpful for making theoretical statements about the NPF.
It also helps us contrast the NPF with another classical machine learning method for stochastic process prediction, _Gaussian processes_ (GPs), which do not incorporate meta-learning (illustrated in the top right quadrant of {numref}`meta_learning_sp`). In this tutorial, we use GPs mainly to benchmark the NPF and provide synthetic datasets.
In contrast to GP prediction, NPs use the expressivity of deep neural networks in their mapping.
In order to do this, each member of the NPF has to address these questions: 1) How can we use neural networks to parameterise a map from datasets to predictive distributions over arbitrary target sets? 2) How can we learn this map?

```{admonition} Note$\qquad$Gaussian Processes
---
class: note, dropdown
---
{numref}`ConvLNP` also shows the predictive mean and error-bars of the ground truth _Gaussian process_ (GP) used to generate the data.
Unlike NPs, GPs require the user to specify a kernel function to model the data.
GPs are attractive due to the fact that exact prediction in GPs can be done _in closed form_.
However, this has computational complexity $\mathcal{O}(N^3)$ in the dataset size, which limits the application of exact GPs to large datasets.
Many accessible introductions to GPs are available online.
Some prime examples are [Distill's visual exploration](https://distill.pub/2019/visual-exploration-gaussian-processes/), [Neil Lawrence's post](http://inverseprobability.com/talks/notes/gaussian-processes.html), or [David Mackay's video lecture](https://www.youtube.com/watch?v=NegVuuHwa8Q).

We note that there is a close relationship between GPs and NPs.
Recall that given an appropriate kernel function $k$, $\mathcal{C}$, and any collection of target inputs $\mathbf{x}_{\mathcal{T}}$, a GP defines the following posterior predictive distribution:

$$
\begin{align}
p (\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) &=
\mathcal{N} \left( \mathbf{y}_{\mathcal{T}}; \mu_{\text{post}}, \Sigma_{\text{post}} \right), \\
\mu_{\text{post}} & = K_{\mathcal{T}, \mathcal{C}} K_{\mathcal{C}, \mathcal{C}}^{-1} \mathbf{y}_{\mathcal{C}}, \\
\Sigma_{\text{post}} & = K_{\mathcal{T}, \mathcal{T}} - K_{\mathcal{T}, \mathcal{C}} K_{\mathcal{C}, \mathcal{C}}^{-1} K_{\mathcal{C}, \mathcal{T}},
\end{align}
$$

where we denote $K_{\mathcal{T}, \mathcal{C}}$ as the matrix constructed by evaluating the kernel at the inputs of the target set with those of the context set, and similarly for $K_{\mathcal{C}, \mathcal{C}}$, $K_{\mathcal{C}, \mathcal{T}}$, and $K_{\mathcal{T}, \mathcal{T}}$.
Thus, we can think of posterior inference in GPs as a map from context sets $\mathcal{C}$ to distributions over predictive functions, just as for NPs.

In this tutorial, we mainly use GPs to specify simple synthetic stochastic processes for benchmarking.
We then train NPs to perform GP prediction.
Since we can obtain the _exact_ GP predictive distribution in closed form, we can use GPs to test the efficacy of NPs.
It is important to remember, however, that NPs can model a much broader range of SPs than GPs can (e.g. natural images, SPs with multimodal marginals), are more computationally efficient at test time, and can learn directly from data without the need to hand-specify a kernel.
```


(parametrizing_npf)=
## Parameterising Neural Processes

Here we will discuss how to parametrise NPs.
As a reminder, here are three design goals that we want each NP to satisfy:

1. **Use neural networks.** The map from context sets to predictive distributions should be parametrized by a neural network. We will use a subscript $\theta$ to denote all the parameters of the network, $p_{\theta}(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}; \mathcal{C})$.
2. **The context set $\mathcal{C}$ should be treated as a set.** This differs from standard vector-valued inputs in that: i) a set may have _varying sizes_; ii) a set has no intrinsic ordering. The second point means that NPs should be _permutation invariant_, i.e., $p_{\theta}(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}; \mathcal{C}) = p_{\theta}(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}; \pi(\mathcal{C}))$ for any permutation operator $\pi$.
3. **Consistency.** The resulting predictive distributions $p_{\theta}(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}; \mathcal{C})$ should be consistent with each other for varying $\mathbf{x}_{\mathcal{T}}$ to ensure that NPs give rise to proper stochastic processes --- see advanced dropdown above for more details.


To satisfy these requirements, NPs first map the entire context set to a representation, $R$, using an encoder $\mathrm{Enc}_{\theta}$.
Specifically, the encoder is always going to be of form $\mathrm{Enc}_{\theta}(\mathcal{C}) = \rho \left ( \sum_{c=1}^C  \phi(x^{(c)}, y^{(c)}) \right)$ for appropriate $\rho$ and $\phi$, which are defined using neural networks.
The sum operation in the encoder is key as it ensures permutation invariance --- due to the commutativity of the sum operation --- and that the resulting $R$ "lives" in the same space regardless of the number of context points $C$.

After that, the NPF splits into two sub-families depending on whether or not the representation is used to define a stochastic latent variable. These sub-families are called the _conditional_ Neural Process family (CNPF), and the _latent_ Neural Process family (LNPF):

* In the CNPF, the predictive distribution at any set of target inputs $\mathbf{x}_{\mathcal{T}}$ is _factorised_ conditioned on $R$. That is, $p_{\theta}(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C}) = \prod_{t=1}^T p_{\theta}(y^{(t)} | x^{(t)}, R)$.

* In the LNPF, the encoding $R$ is used to define a global _latent variable_ $\mathbf{z} \sim p_{\theta}(\mathbf{z} | R)$. The predictive distribution is then factorised _conditioned on_ $\mathbf{z}$. That is, $p_{\theta}(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C}) = \int \prod_{t=1}^T p_{\theta}(y^{(t)} | x^{(t)}, \mathbf{z}) p_{\theta}(\mathbf{z} | R) \, \mathrm{d}\mathbf{z}$.

We will call the decoder, $\mathrm{Dec}_{\theta}$, the map parametrizing the predictive distribution using the target input $x^{(t)}$ and the encoding of the context set --- $R$ in the case of the CNPF and (samples of) $z$ for the LNPF.
Typically the predictive distribution is Gaussian meaning that the decoder predicts a mean $\mu^{(t)}$ and a variance $\sigma^{2(t)}$.
As we will later show, it is the factorisation assumptions in the decoder that ensure our consistency requirement.


````{admonition} Concrete Example
---
class: note
---
As a concrete example of what a Neural Process looks like, {numref}`CNP_gif` shows a schematic animation of the forward pass of a _Conditional Neural Process_ (CNP), the simplest member of the CNPF.
We see that every $(x, y)$ pair in the context set (here with three datapoints) is passed through an MLP $e$ to obtain a local encoding.
The local encodings $\{r_1, r_2, r_3\}$ are then aggregated by a mean pooling $a$ to a representation $r$.
Finally, the representation $r$ is fed into another MLP $d$ along with the target input to yield the mean and variance of the predictive distribution of the target output $y$.
We'll take a much more detailed look at the CNP {ref}`later <cnp>`.

```{figure} ../gifs/NPFs.gif
---
width: 25em
name: CNP_gif
alt: Schematic representation of CNP forward pass.
---
Schematic representation of CNP forward pass taken from [Marta Garnelo](https://www.martagarnelo.com/conditional-neural-processes).
```
````

The forward pass for members of both the CNPF and LNPF is represented schematically in {numref}`computational_graph_NPFs`.
For the LNPF there is an extra step of sampling the latent variable $\mathbf{z}$ in between $R$ and $\mathrm{Dec}_{\theta}$.[^det]

```{figure} ../images/computational_graph_NPFs.png
---
width: 30em
name: computational_graph_NPFs
alt: high level computational graph of NPF
---
High level computational graph of the Neural Process Family.
```

As we'll see in the following pages of this tutorial, the CNPF and LNPF come with their own advantages and disadvantages.
Roughly speaking, the LNPF allows us to model _dependencies_ in the predictive distribution over the target set, at the cost of requiring us to approximate an intractable objective function.

Furthermore, even _within_ each family, there are myriad choices that can be made. The most important is the choice of encoder architecture.
Each of these choices will lead to neural processes with different inductive biases and capabilities.
As a teaser, we provide a very brief summary of the neural processes considered in this tutorial (This should be skimmed for now, but feel free to return here to get a quick overview once each model has been introduced. Clicking on each model brings you to the Reproducibility page which includes code for running the model):

```{list-table} Summary of different members of the Neural Process Family
---
header-rows: 1
stub-columns: 1
name: summary_npf
---

* - Model
  - Encoder
  - Spatial generalisation
  - Predictive fit quality
  - Additional Assumption
* - {doc}`Conditional NP <../reproducibility/CNP>`[^CNP], {doc}`Latent NP <../reproducibility/LNP>`[^LNP]
  - MLP + Mean-pooling
  - No
  - Underfits
  - None
* - {doc}`Attentive CNP <../reproducibility/AttnCNP>`[^AttnCNP], {doc}`Attentive LNP <../reproducibility/AttnLNP>`[^AttnLNP]
  - MLP + Attention
  - No
  - Less underfitting, jagged samples
  - None
* - {doc}`Convolutional CNP <../reproducibility/ConvCNP>`[^ConvCNP], {doc}`Convolutional LNP <../reproducibility/ConvLNP>`[^ConvLNP]
  - SetConv + CNN + SetConv
  - Yes
  - Less underfitting, smooth samples
  - Translation Equivariance
```

In the {doc}`CNPF <CNPF>` and {doc}`LNPF <LNPF>` pages of this tutorial, we'll dig into the details of how all these members of the NPF are specified in practice, and what these terms really mean.
For now, we simply note the range of options and tradeoffs.
To recap, we've (schematically!) thought about how to parameterise a map from observed context sets $\mathcal{C}$ to predictive distributions at any target inputs $\mathbf{x}_{\mathcal{T}}$ with neural networks.
Next, we consider how to _train_ such a map, i.e. how to learn the parameters $\theta$.

(meta_training)=
## Meta-Training in the NPF

To perform _meta_-learning, we require a _meta_-dataset or a _dataset of datasets_.
In the meta-learning literature, each dataset in the meta-dataset is referred to as a _task_.
For the NPF, this means having access to many independent samples of functions from the data-generating process.
Each sampled function is then a task. We would like to use this meta-dataset to learn how to make predictions at a target set upon observing a context set. To do this, we use an _episodic training procedure_, common in meta-learning. Each episode can be summarised in five steps:
<!-- For example, we may have a large collection of audio waveforms $\mathcal{M} := \{ \mathcal{D}_i \}_{i=1}^{N_{\mathrm{tasks}}}$ from different speakers.
Each of these waveforms is itself a time-series $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$, where each $x^{(n)}, y^{(n)}$ is a timestamp/audio amplitude pair.
Or we might have a large collection of natural images.
Then each $\mathcal{D}$ would be a single image consisting of pixel-location/pixel-value pairs. -->



1. Sample a task $\mathcal{D}$ from $\{ \mathcal{D}_i \}_{i=1}^{N_{\mathrm{tasks}}}$.
2. Randomly split the task into context and target sets: $\mathcal{D} = \mathcal{C} \cup \mathcal{T}$.
3. Pass $\mathcal{C}$ through the Neural Process to obtain the predictive distribution at the target inputs, $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})$.
4. Compute the log likelihood $\mathcal{L} = \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})$ which measures the predictive performance on the target set.[^objective]
Note that for the LNPF, we will have to compute an approximation or a lower bound of the log-likelihood objective.
5. Compute the gradient $\nabla_{\theta}\mathcal{L}$ for stochastic gradient optimisation.

The episodes are repeated until training converges.
Intuitively, this procedure encourages the NPF to produce predictions that fit an unseen target set, given access to only the context set.
Once meta-training is complete, if the Neural Process generalises well, it will be able to do this for brand new, unseen context sets.
To recap, we've seen how the NPF can be thought of as a family of meta-learning algorithms, taking entire datasets as input, and providing predictions with a single forward pass.

```{admonition} Advanced$\qquad$Maximum-Likelihood Training
---
class: hint, dropdown
---

To better understand the objective $\mathcal{L}(\theta ; \mathcal{C}, \mathcal{T} ) = \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})$ ---which we refer to as _maximum-likelihood_ training--- let $p(\mathcal{D}) = p(\mathcal{C}, \mathcal{T})$ be the _task distribution_, that is, the distribution from which we sample tasks in the episodic training procedure.
Then, we can express the meta-objective as an expectation over the task distribution:

$$
\begin{align}
\mathbb{E}_{p(\mathcal{D})} [\mathcal{L}(\theta ;\mathcal{C}, \mathcal{T})] &= \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}} , \mathbf{y}_{\mathcal{T}} )} [\log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})] \\
&= \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}})} \left[  \mathbb{E}_{p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})}  [ \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C}) ]  \right] \\
&= - \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}})} \left[ \mathrm{KL} (p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C}) \| p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})  ) \right] + \mathrm{const.}
\end{align}
$$

Here $\mathrm{KL}$ is the KL-divergence, which is a measure of how different two distributions are, and $\mathrm{const.}$ is a constant that is independent of $\theta$.
Hence we can see that maximising the meta-learning objective is equivalent to _minimising the task averaged KL-divergence_ between the NPF predictive $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}; \mathcal{C})$ and the conditional distribution $p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})$ of the task distribution.

This point of view also helps us see the potential issues with maximum-likelihood training if $p(\mathcal{D})$ is not truly representative of the data-generating process.
For example, if all of the tasks sampled from $p(\mathcal{D})$ have their $x$-values bounded in a range $[-a, a]$, then the expected KL will not include any contributions from $p_\theta(y| x; \mathcal{C})$ when $x$ is outside of that range.
Thus there is no direct incentive for the algorithm to learn a reasonable predictive distribution for that value of $x$.
To avoid this problem, we should make our meta-dataset as representative of the tasks we expect to encounter in the future as possible.

Another way of addressing some of these issues is to bake appropriate _inductive biases_ into the Neural Process.
In the next page of this tutorial, we will see that Convolutional Neural Processes use _translation equivariance_ to make accurate predictions for $x$-values outside the training range when the underlying data-generating process is stationary.


```

<!-- ## Stochastic Process Prediction

Let's look back to {numref}`ConvLNP`. Intuitively speaking, we can think of the context points as being observations of an _unknown, random function_. The solid blue lines of the NP are then samples from the predictive distribution of that function _conditioned on_ the observed context points. In mathematical terms, a probability distribution over a random function is called a _stochastic process_. This is where the name Neural Process comes from! The NPF is a method of using neural networks to specify a distribution over a random function.

How does this relate to the description we gave earlier? In the previous section we thought of the NPF as returning a predictive distribution $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ given a context set $\mathcal{C}$. However, in this expression, we are free to choose $\mathbf{x}_{\mathcal{T}}$ to be any target inputs we like. If we then choose the target inputs to be _the entire real line_, we are effectively specifying a distribution over $y(x)$ _for all_ $x$ --- in other words, a random function, i.e. a stochastic process. Hence the NPF can be thought of as learning a map from observed context sets to predictive stochastic processes.

This point of view of the NPF can provide good intuition and is helpful for making theoretical statements about the NPF. It also helps us contrast the Neural Process Family with another classical machine learning method for stochastic process prediction, _Gaussian processes_.

There are, however, several technical conditions that need to be satisfied to ensure that the NPF specifies a genuinely consistent stochastic process. These conditions are prominent in the original literature on Neural Processes, and help motivate some of the design choices in the NPF, such as the factorisation assumption in the CNPF. We provide a discussion of some of these issues in the dropdown box below. However, it is possible to follow the rest of this tutorial without covering this more technical section, so **feel free to skim or skip it on a first reading**. In summary, for a fixed context set $\mathcal{C}$, the NPF predictive distribution indeed defines a single consistent predictive stochastic process, although this may not be the case for varying context sets. -->

<!-- ## Use Cases

What tasks can a trained Neural Process be applied to?
Broadly speaking, they can be used in any situation where predictions need to be made under uncertainty.
We list a few examples here.

* **Regression with uncertainty**. Imagine you're writing an algorithm to predict rainfall in the future, or to forecast demand for materials in your supply chain. These are all regression tasks where a well-trained Neural Process could exploit intricate structure in the data to make predictions and provide crucial uncertainty estimates.

* **Interpolating missing values**. This is closely related to the previous use case. Given a corrupted image or audio signal, we may want to see what the plausible interpolations are. Moreover, if the corrupted region is large, we want to be aware of the whole range of plausible interpolations, not just one guess. If the image is, say, a medical scan, this could be crucial for decision-making.

* **Active learning**. In active learning, our goal is to provide accurate predictions with as few measurements as possible. This is typically done by performing measurements at points with the greatest uncertainty. A Neural Process can be used to provide these uncertainty estimates. Once the measurement is taken, the new context point can be fed back into the Neural Process, and the uncertainty estimates can be updated for the next measurement.

* **Bayesian optimisation**. In Bayesian optimisation, the goal is to perform black-box optimisation of an unknown function when gradient information is unavailable, and each evaluation of the objective function is expensive. Most algorithms for Bayesian optimisation rely on querying points which have high expected reward and also high uncertainty, thus trading off exploration and exploitation. -->

## Summary of NPF Properties

Would an NPF be a good fit for your machine learning problem? To summarise, we note the advantages and disadvantages of the NPF:

* &#10003; **Fast predictions** on new context sets at test time. Often, training a machine learning model on a new dataset is computationally expensive. However, meta-learning allows the NPF to incorporate information from a new context set and make predictions with a _single_ forward pass. Typically the complexity will be linear or quadratic in the context set size instead of cubic as with standard Gaussian process regression.
* &#10003; **Well calibrated uncertainty**. Often meta-learning is applied to situations where each task has only a small number of examples at test time (also known as _few-shot learning_). These are exactly the situations where we should have uncertainty in our predictions, since there are many possible ways to interpolate a context set with few points. The NPF _learns_ to represent this uncertainty during episodic training.
* &#10003; **Data-driven expressivity**. The enormous flexibility of deep learning architectures means that the NPF can learn to model very intricate predictive distributions directly from the data. The user mainly has to specify the inductive biases of the network architecture, e.g. convolutional vs attentive.

However, these advantages come at the cost of the following disadvantages:

* &#10007; **The need for a large dataset for meta-training**. Meta-learning requires training on a large dataset of target and context points sampled from different functions, i.e., a large dataset of datasets. In some situations, a dataset of datasets may simply not be available. Furthermore, although predicting on a new context set after meta-training is fast, meta-training itself can be computationally expensive depending on the size of the network and the meta-dataset.

* &#10007; **Underfitting and smoothness issues**. The NPF predictive distribution has been known to underfit the context set, and also sometimes to provide unusually jagged predictions for regression tasks. The sharpness and diversity of the image samples for the LNPF could also be improved. However, improvements are being made on this front, with both the attentive and convolutional variants of the NPF providing significant advances.

In summary, we've taken a bird's eye view of the Neural Process Family and seen how they specify a map from datasets to stochastic processes, and how this map can be trained via meta-learning. We've also seen some of their use-cases and properties. Let's now dive into the actual architectures! In the next two pages we'll cover everything you need to know to get started with the models in Conditional and Latent Neural Process Families.


[^det]: The general computational graph of the NPF actually has a latent variable $\mathbf{z}$. Indeed, the CNPF may be thought of as the LNPF in the case when the latent variable $\mathbf{z}$ is constrained to be deterministic ($p_{\theta}(\mathbf{z} | R)$ is a Dirac delta function).

[^objective]: During training the performance is usually measured on both the context and target set, i.e. we append the context set to the target set.

[^CNP]: {cite}`garnelo2018conditional`.

[^LNP]: {cite}`garnelo2018neural` --- in this paper and elsewhere in the Neural Process literature, the authors refer to latent neural processes simply as neural processes. In this tutorial we use the term "neural process" to refer to both conditional neural processes and latent neural processes. We reserve the term "latent neural process" specifically for the case when there is a stochastic latent variable $\mathbf{z}$.

[^AttnCNP]: {cite}`kim2019attentive` --- this paper only introduced the latent variable Attentive LNP, but one can easily drop the latent variable to obtain the Attentive CNP.

[^AttnLNP]: {cite}`kim2019attentive`.

[^ConvCNP]: {cite}`gordon2019convolutional`.

[^ConvLNP]: {cite}`foong2020convnp`.
