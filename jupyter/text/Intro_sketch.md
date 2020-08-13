# The Neural Process Family

In this tutorial, we introduce the **Neural Process Family** (NPF). Neural Processes (NPs) are a family of deep learning methods that use **meta-learning** to model **stochastic processes**. They are particularly well-suited to tasks which require fast test-time inference and well-calibrated uncertainty, and have been applied to time-series data, image data and environmental data, among others. The goal of this tutorial is to provide a gentle introduction to the NPF. Our approach is to walk through several prominent members of the NPF, highlighting their key advantages and drawbacks. By accompanying the exposition with code and examples, we hope to (i) make the design choices and tradeoffs associated with each model clear and intuitive, and (ii) demonstrate both the simplicity of the framework and its broad applicability.

% I don't know how to put them side by side :/

```{figure} ../gifs/ConvLNP_norbf_gp_extrap.gif
---
width: 35em
name: ConvLNP
alt: Samples from ConvLNP trained on GPs
---
Sample functions from the predictive distribution of ConvLNPs (blue) and the oracle GP (green) with periodic and noisy Matern kernels.
```

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_intro
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```

## Outline and Motivation

This tutorial is split into three sections. This first page will give a broad, bird's eye view of the entire Neural Process Family. We'll see that the family splits naturally into two sub-families: the Conditional Neural Process Family (CNPF), and the Latent Neural Process Family (LNPF). These are covered in the following two pages, and there we'll get into more detail about the different architectures in each family. We'll make liberal use of dropdown boxes in this tutorial to avoid breaking up the exposition. Feel free to skip or skim the boxes marked as 'advanced' on a first reading. Without further ado, let's get started!

```{admonition} Note
---
class: note
---
**All the code to run the models discussed in this tutorial is provided** --- these can be accessed in the Reproducibility section. (LINK TO GITHUB REPO?)
```

To motivate the NPF, consider the following tasks:

- Predicting **time-series data**. We are given a dataset $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$, where $x$ are the inputs (time) and $y$ are the outputs. We may want to predict the distribution of a new output $p(y^{(* )}|x^{(* )}, \mathcal{D})$ at any test location $x^{(* )}$, or to sample an entire time-series conditioned on $\mathcal{D}$. For example, these could be audio waveforms, and $y$ might be the sound wave amplitude. In a corrupted audio signal, we may only have $\mathcal{D}$ measured at a sparse set of points on the $x$-axis. We then want to make predictions that interpolate and extrapolate missing values of $y$. In regions of the $x$-axis with little data, there could be many reasonable predictions --- hence we don't just want point estimates but also measures of **uncertainty**. Ideally, we could also sample the entire time-series, so we could listen to many plausible completions of the data. {numref}`ConvLNP` shows an example of the NPF being used to sample plausible interpolations of simple time-series, both periodic and non-periodic.

- Interpolating **image data** with uncertainty estimates. For example, we may be given satellite images of a region, which may be obscured by cloud-cover, or a medical imaging scan which has some occluded regions. In both cases we might be interested not just in a single interpolation, but in the _entire probability distribution_ over them. Images can be viewed as real-valued _functions_ on the two-dimensional plane. Each input $x$ is a two-dimensional vector denoting pixel location in the image, and each $y$ is a real number representing pixel intensity (or a three-dimensional vector for RGB images). {numref}`ConvCNP_superes_intro` shows the NPF being used to interpolate missing pixels in images of faces.

Each member of the NPF is a method for tackling problems such as these. To do this, the NPF brings together two key ideas: **meta-learning** and **stochastic process prediction**.

## Meta-learning

What is meta-learning? Simply put, meta-learning is _learning to learn_. In a standard machine learning algorithm, a neural network is trained (e.g. by stochastic gradient descent) on a single dataset $\mathcal{D}$, and returns a predictor, $f(\cdot)$. At test time, a prediction at $x^{(* )}$ can be made by feeding it into the predictor to obtain $f(x^{(* )})$. By contrast, a meta-learning algorithm is trained to _directly_ model a mapping from datasets to predictions. To be concrete, let's say we have some observed data $\mathcal{C} = \{(x^{(c)}, y^{(c)})\}_{c=1}^C$. In the NPF literature, this is referred to as a _context set_. We would now like to make predictions for the value of $y$ at some query locations $\{x^{(t)}\}_{t=1}^T = \mathbf{x}_{\mathcal{T}}$, conditioned on $\mathcal{C}$. In the NPF literature, $\mathbf{x}_{\mathcal{T}}$ is referred to as the _target inputs_ or _target features_, and together with their corresponding outputs $\{y^{(t)}\}_{t=1}^T = \mathbf{y}_{\mathcal{T}}$, the pair $\mathcal{T} = \{(x^{(t)}, y^{(t)})\}_{t=1}^T$ is referred to as the _target set_.

Whereas a standard neural network would have to be trained on each new context set $\mathcal{C}$, and then queried at the target inputs $\mathbf{x}_{\mathcal{T}}$, a meta-trained neural network would simply take $\mathcal{C}$ as an input and return the predictions $f(\mathbf{x}_{\mathcal{T}})$ _with a single forward pass_, without any gradient updates! (Here $f(\mathbf{x}_{\mathcal{T}})$ is a vector containing $f(x)$ for all $x \in \mathbf{x}_{\mathcal{T}}$.)

When would such a scheme be useful in practice? We'll soon see that meta-learning is applied in scenarios where we have access to a large collection $\{ \mathcal{D_i} \}_{i=1}^{N_{\mathrm{tasks}}}$ of related datasets (called _tasks_ in the meta-learning terminology). This collection of tasks forms the _meta-dataset_, or dataset of datasets, on which the meta-learning algorithm is trained. At meta-test time, we'll present the algorithm with a task it's never seen before. It will then use the information obtained during meta-learning to make predictions on this new task --- in other words, meta-learning _shares information across tasks to make predictions_.

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

* - $\{ \mathcal{D_i} \}_{i=1}^{N_{\mathrm{tasks}}}$
  - Meta-dataset
* - $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$
  - Dataset/task
* - $\mathcal{C} = \{(x^{(c)}, y^{(c)})\}_{c=1}^C$
  - Context set
* - $\mathbf{x}_{\mathcal{T}} = \{x^{(t)}\}_{t=1}^T$
  - Target inputs
* - $\mathbf{y}_{\mathcal{T}} = \{y^{(t)}\}_{t=1}^T$
  - Target outputs
* - $\mathcal{T} = (\mathbf{x}_{\mathcal{T}}, \mathbf{y}_{\mathcal{T}}) = \{(x^{(t)}, y^{(t)})\}_{t=1}^T$
  - Target set
* - $f(\mathbf{x}_{\mathcal{T}}) = \{ f(x) \}_{x \in \mathbf{x}_{\mathcal{T}}}$
  - Predictions
```

Because it shares information across tasks, meta-learning is especially well-suited to situations where each task is a _small_ dataset (by deep learning standards), as in, e.g., few-shot learning. However, this raises the question: if the context set is small, can we really expect to obtain a unique predictor $f(\cdot)$ from it? To relate this back to our examples, if we only observe an audio signal at a few timestamps, or an image at a few pixels, can we really uniquely reconstruct the original? What we need is to express our _uncertainty_, and this leads us naturally to _stochastic processes_.

<!-- a meta-learning algorithm is trained to _directly_ model the map from datasets to predictive distributions. To be concrete, let's say we wish to make predictions conditioned on some observed data $\mathcal{C} = \{(x^{(c)}, y^{(c)})\}_{c=1}^C$. In the NPF literature, this is referred to as a _context set_. We would now like to make predictions for the value of $y$ at some query locations $\{x^{(t)}\}_{t=1}^T = \mathbf{x}_{\mathcal{T}}$. In the NPF literature, $\mathbf{x}_{\mathcal{T}}$ is referred to as the _target inputs_ or _target features_, and together with their corresponding outputs $\{y^{(t)}\}_{t=1}^T = \mathbf{y}_{\mathcal{T}}$, the pair $\mathcal{T} = \{(x^{(t)}, y^{(t)})\}_{t=1}^T$ is referred to as the _target set_.

Whereas a standard neural network would have to be trained on each new context set $\mathcal{C}$ (e.g. by stochastic gradient descent), and then queried at the target inputs $\mathbf{x}_{\mathcal{T}}$, a meta-trained Neural Process simply takes $\mathcal{C}$ as an input and returns the predictive distribution at $\mathbf{x}_{\mathcal{T}}$ _in a single forward pass_, without any gradient updates! In order to do this, each member of the NPF has to address these questions: 1) How can we parameterise a map from datasets to predictive predictive distributions over arbitrary target sets? 2) How can we learn this map? -->

## Stochastic Process Prediction

We've seen that we can think of meta-learning as learning a map directly from context sets $\mathcal{C}$ to predictor functions $f(\cdot)$. However, there are many situations where a single predictor without error-bars isn't good enough. Quantifying our uncertainty is crucial for decision-making (think of a doctor relying on an algorithm to diagnose a patient), and has many applications in active learning, Bayesian optimisation, and more. It is especially crucial in the small context-set setting, which is often the case in meta-learning.

What we need is not a single predictor, but a _distribution over predictors_. Each sample from this distribution would represent a plausible interpolation of the data, and the diversity of the samples would reflect the _uncertainty_ in our predictions --- think back to {numref}`ConvLNP`. Since each predictor $f(\cdot)$ is a function, this is a _distribution over functions_. In mathematics, this is known as a _stochastic process_ (SP). Hence, **the NPF can generally be seen as deploying and training neural networks to learn maps from datasets to predictive stochastic processes**. This is where the name Neural Process comes from!

How can we specify an SP? Previously, we thought of the predictor as mapping any target inputs $\mathbf{x}_{\mathcal{T}}$ to $f(\mathbf{x}_{\mathcal{T}})$, given a context set $\mathcal{C}$. Now, for any $\mathcal{C}$, and any choice of target inputs $\mathbf{x}_{\mathcal{T}}$, the NPF will return a _probability distribution_ $p(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}, \mathcal{C})$. As long as these distributions are consistent with each other for different choices of $\mathbf{x}_{\mathcal{T}}$, specifying $p(\mathbf{y}_{\mathcal{T}}| \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ for all finite sets of target inputs $\mathbf{x}_{\mathcal{T}}$ is equivalent to specifying an SP.

```{admonition} Advanced$\qquad$Stochastic Process Consistency
---
class: hint, dropdown
---
In the previous discussion, we considered specifying a stochastic process (SP) by specifying $p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ for all finite collections of target inputs $\mathbf{x}_{\mathcal{T}}$. Each  distribution for a given $\mathbf{x}_{\mathcal{T}}$ is referred to as a _finite marginal_ of the SP. Can we stitch together all of these finite marginals to obtain a single SP? The _Kolmogorov extension theorem_ tells us that we can, as long as the marginals are consistent with each other under _permutation_ and _marginalisation_:

To illustrate these consistency conditions, let's look at some artificial examples of finite marginals that are _not_ consistent. Let $x^{(1)}, x^{(2)}$ be two target inputs, with $y^{(1)}, y^{(2)}$ the corresponding random outputs.

1. Consider a collection of finite marginals with $y^{(1)} \sim \mathcal{N}(0, 1)$ and $[y^{(1)}, y^{(2)}] \sim \mathcal{N}([10, 0], \mathbf{I})$. What is the mean of $y^{(1)}$?
2. Consider a collection with $[y^{(1)}, y^{(2)}] \sim \mathcal{N}([0, 0], \mathbf{I})$ and $[y^{(1)}, y^{(2)}] \sim \mathcal{N}([1, 1], \mathbf{I})$. What is the mean of $y^{(1)}$? What is the mean of $y^{(2)}$?

We see the problem here: inconsistent marginals lead to self-contradictory predictions! In the first example, the marginals were not _consistent under marginalisation_: marginalising out $y^{(2)}$ from the distribution of $[y^{(1)}, y^{(2)}]$ did not yield the distribution of $y^{(1)}$. In the second case, the marginals were not _consistent under permutation_: the distributions differed depending on whether you considered $y^{(1)}$ first or $y^{(2)}$ first. Later we'll prove that this problem will never occur for members of the NPF --- for a fixed context set $\mathcal{C}$, the NPF predictive distributions always define a consistent stochastic process. This is where the name Neural Process comes from!

<!-- Take some time to convince yourself that, _for a fixed context set_ $\mathcal{C}$, this problem will never occur for the CNPF or the LNPF. For example, for the CNPF, we have $p_{\theta}(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \prod_{t=1}^T p_{\theta}(y^{(t)} | x^{(t)}, R)$. Hence for any two target inputs,

$$
\begin{align}
\int p_{\theta} (y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}, \mathcal{C}) \, \mathrm{d}y^{(2)} &= \int p_{\theta} (y^{(1)}| x^{(1)}, R) p_{\theta} (y^{(2)}| x^{(2)}, R) \, \mathrm{d}y^{(2)} \\
&= p_{\theta} (y^{(1)}| x^{(1)}, R) \int  p_{\theta} (y^{(2)}| x^{(2)}, R) \, \mathrm{d}y^{(2)}\\
&= p_{\theta} (y^{(1)}| x^{(1)}, R),
\end{align}
$$

which is exactly the definition of the predictive distribution if we had considered $x^{(1)}$ on its own! Hence we see that the factorisation assumption automatically leads to consistency under marginalisation. You can verify that the same is true for the LNPF. -->

So far, we've only considered what happens when you fix the context set and vary the target inputs. There is one more kind of consistency that we might expect SP predictions to satisfy: consistency among predictions with _different context sets_. Consider two input-output pairs, $(x^{(1)}, y^{(1)})$ and $(x^{(2)}, y^{(2)})$. The product rule of probability tells us that the joint predictive density must satisfy

$$
\begin{align}
p(y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}) &= p(y^{(1)}| x^{(1)}) p(y^{(2)}| y^{(1)}, x^{(1)}, x^{(2)}) \\
&= p(y^{(2)}| x^{(2)}) p(y^{(1)}| y^{(2)}, x^{(1)}, x^{(2)}).
\end{align}
$$

This essentially states that the distribution over $y^{(1)}, y^{(2)}$ obtained by autoregressive sampling should be independent of the order in which the sampling is performed. Unfortunately, this is _not_ guaranteed to be the case for any members of the NPF. In practice, the NPF yields good predictive performance even though it violates this consistency. This is because a well-trained Neural Process _learns_ a map that is approximately consistent in this way, even though it is not explicitly constrained to.

```

This point of view of the NPF can provide good intuition and is helpful for making theoretical statements about the NPF. It also helps us contrast the Neural Process Family with another classical machine learning method for stochastic process prediction, _Gaussian processes_ (GPs), which we use mainly as a benchmark.

```{admonition} Note$\qquad$Gaussian Processes
---
class: note, dropdown
---
{numref}`ConvLNP` also shows the predictive mean and error-bars of the ground truth _Gaussian process_ (GP) used to generate the data. Unlike the NPF, GPs require the user to specify a kernel function to model the data. GPs are attractive due to the fact that exact prediction in GPs can be done _in closed form_. However, this has computational complexity $\mathcal{O}(N^3)$ in the dataset size, which limits the application of exact GPs to large datasets. For an accessible introduction to GPs, see INSERT REF.

In this tutorial, we mainly use GPs to specify simple synthetic stochastic processes for benchmarking. We then train NPs to perform GP prediction. Since we can obtain the _exact_ GP predictive distribution in closed form, we can use GPs to test the efficacy of NPs. It is important to remember, however, that NPs can model a much broader range of SPs than GPs can (e.g. natural images, SPs with multimodal marginals), are more computationally efficient at test time, and can learn directly from data without the need to hand-specify a kernel.   
```

In contrast to GP prediction, in Neural Processes we want to make use of the expressivity of deep neural networks in our mapping. In order to do this, each member of the NPF has to address these questions: 1) How can we use neural networks to parameterise a map from datasets to predictive distributions over arbitrary target sets? 2) How can we learn this map?

```{admonition} Advanced$\qquad$Relation to MAML
---
class: hint, dropdown
---
TO BE WRITTEN

```

## Parameterising the Predictor

Let's first consider how to parameterise a map directly from datasets to predictive distributions using deep neural networks. The first challenge we have to address is how to input a _dataset_ into a neural network. This differs from standard vector-valued inputs in two ways:

* Datasets may have _varying sizes_: we might condition on a context set of size 1, 2 or 100. We would like the same architecture to be able to handle all these cases.
* Datasets have no intrinsic ordering. Hence any map $E$ acting on a dataset $\mathcal{C}$ should be _permutation invariant_: $E(\mathcal{C}) = E(\pi (\mathcal{C}))$, where $\pi$ is any permutation operator.

```{admonition} Advanced$\qquad$Deep Sets
---
class: hint, dropdown
---
In {cite}`zaheer2017deep`, the authors show that (subject to certain conditions) any permutation invariant map $M$ from a sequence $( x^{(n)} )_{n=1}^N$ to a real number can be expressed as:

$$
\begin{align}
M \left(( x^{(n)} )_{n=1}^N \right) = \rho \left ( \sum_{n=1}^N \phi(x^{(n)}) \right),
\end{align}
$$

for some suitable functions $\rho$ and $\phi$. This is known as a 'sum decomposition' or a 'Deep Sets encoding'. The NPF makes heavy use of sum-decompositions in its architectures. This result tells us that, as long as $\rho$ and $\phi$ are universal function approximators, this sum-decomposition can be done without loss of generality in terms of the class of permutation-invariant maps that can be expressed. We will later encounter an extension of this result for _translation equivariant_ functions, proven in {cite}`gordon2019convolutional`.

```

To satisfy these requirements, members of the NPF first encode each point in the context set $D_c$ separately: $R^{(c)} = e_{\theta}(x^{(c)}, y^{(c)})$ for each $(x^{(c)}, y^{(c)}) \in D_c$. Here $R^{(c)}$ can be thought of as a local encoding of the datapoint --- one for each point in the context set. $e_{\theta}$ is a local encoding function, and can be represented with a deep neural network (we use $\theta$ to denote all the parameters in the Neural Process). These local encodings are then combined into a single global encoding $R_{\theta}(\mathcal{C})$ using an aggregation function, $\mathrm{Agg}$. Importantly, $\mathrm{Agg}$ is chosen to be _permutation invariant_: for any permutation $\pi$ of $\{ 1, 2, ..., C \}$,

$$
\mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)=\mathrm{Agg}\left(\pi\left(\{R^{(c)}\}_{c=1}^{C} \right)\right)
$$

```{admonition} Note$\qquad$Mean-pooling
---
class: note, dropdown
---
A simple and popular choice for $\mathrm{Agg}$ is _mean-pooling_:

$$
\mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right) = \frac{1}{C} \sum_{c=1}^C R^{(c)},
$$

which is easily seen to be permutation invariant.
```

The global encoding $R_{\theta}(\mathcal{C}) = \mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)$ can be thought of as a representation of the entire context set. The information in this encoding will then be used by the Neural Process to make predictions at the target set. At this point, depending on what we do next, the NPF splits into two sub-families: the _conditional_ Neural Process family (CNPF), and the _latent_ Neural Process family (LNPF):

* In the CNPF, the predictive distribution at any set of target inputs $\mathbf{x}_{\mathcal{T}}$ is defined to be _factorised_ conditioned on $R = R_{\theta}(\mathcal{C})$. That is, $p_{\theta}(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \prod_{t=1}^T p_{\theta}(y^{(t)} | x^{(t)}, R)$. Furthermore, each term $p_{\theta}(y^{(t)} | x^{(t)}, R)$ in the product is typically chosen to be a simple distribution over $y^{(t)}$, usually a Gaussian.

* In the LNPF, the encoding $R = R_{\theta}(\mathcal{C})$ is used to define a _latent variable_ $\mathbf{z} \sim p_{\theta}(\mathbf{z} | R)$. The predictive distribution is then defined to be factorised _conditioned on_ $\mathbf{z}$. That is, $p_{\theta}(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \int \prod_{t=1}^T p_{\theta}(y^{(t)} | x^{(t)}, \mathbf{z}) p_{\theta}(\mathbf{z} | R) \, \mathrm{d}\mathbf{z}$.

The forward pass for members of both the CNPF and LNPF is represented schematically in {numref}`computational_graph_NPFs`. Here $e$ is often referred to as the _encoder_ and $d$, the operation which maps the representation $R$ together with the target input to produce the prediction at the target output, is called the _decoder_. For the LNPF there would be an extra step of sampling the latent variable $\mathbf{z}$ in between $R$ and $d$.

```{figure} ../images/computational_graph_NPFs.svg
---
width: 300em
name: computational_graph_NPFs
alt: high level computational graph of NPF
---
High level computational graph of the Neural Process Family.
```

```{admonition} Note$\qquad$CNPF as Deterministic LNPF
---
class: note, dropdown
---
The CNPF may be thought of as the LNPF in the case when the latent variable $\mathbf{z}$ is constrained to be deterministic ($p_{\theta}(\mathbf{z} | R)$ is a Dirac delta function).
```

```{admonition} Note$\qquad$Permutation-invariant Predictions
---
class: note, dropdown
---
Assuming that $\mathrm{Agg}$ is permutation-invariant, convince yourself that the predictive distributions of both the CNPF and LNPF are invariant to permutations of the context set --- that is, our predictions at the target set don't change if the context set is simply shuffled around.

```

As a concrete example of what a Neural Process looks like, {numref}`CNP` shows a schematic animation of the forward pass of a _Conditional Neural Process_ (CNP), a member of the CNPF that was the first Neural Process to be introduced, and conceptually the easiest to understand. We see that every $(x, y)$ pair in the context set (here with three datapoints) is passed through an MLP $e$ to obtain a local encoding, which is then aggregated. Finally, the representation is fed into another MLP $d$ along with the target input to yield the mean and variance of the predictive distribution of the target output $y$. We'll take a much more detailed look at the CNP in CNPF page of this tutorial.

```{figure} ../gifs/NPFs.gif
---
width: 30em
name: CNP
alt: Schematic representation of CNP forward pass.
---
Schematic representation of CNP forward pass taken from [Marta Garnelo](https://www.martagarnelo.com/conditional-neural-processes).
```

As we'll see in the following pages of this tutorial, both the CNPF and LNPF come with their specific advantages and disadvantages. Roughly speaking, the LNPF allows us to model _dependencies_ in the predictive distribution over the target set, at the cost of requiring us to approximate an intractable objective function.

Furthermore, even _within_ each family, there are myriad choices that can be made: Should we use an MLP or a convolutional network? What kind of aggregator function should we use? Each of these choices will lead to neural processes with different inductive biases and capabilities. For example, later in the CNPF and LNPF pages of this tutorial, we will see that incorporating attention can help reduce underfitting, and that incorporating convolutions can help with spatial generalisation. As a teaser, we provide a very brief summary of the neural processes considered in this tutorial (This should be skimmed for now, but feel free to return here to get a quick overview once each model has been introduced. Clicking on each model brings you to the Reproducibility page which includes code for running the model):

```{list-table} Summary of different members of the Neural Process Family
---
header-rows: 1
stub-columns: 1
name: summary_npf
---

* - Model
  - Network architecture
  - Aggregator
  - Spatial generalisation
  - Predictive fit quality
* - {doc}`Conditional NP <../reproducibility/CNP>`[^CNP], {doc}`Latent NP <../reproducibility/LNP>`[^LNP]
  - MLP
  - Mean-pooling
  - No
  - Underfits
* - {doc}`Attentive CNP <../reproducibility/AttnCNP>`[^AttnCNP], {doc}`Attentive LNP <../reproducibility/AttnLNP>`[^AttnLNP]
  - MLP and self-attention
  - Cross-attention
  - No
  - Less underfitting, jagged samples
* - {doc}`Convolutional CNP <../reproducibility/ConvCNP>`[^ConvCNP], {doc}`Convolutional LNP <../reproducibility/ConvLNP>`[^ConvLNP]  
  - Convolutional
  - Set-convolution
  - Yes
  - Less underfitting, smooth samples
```

In the CNPF and LNPF pages of this tutorial, we'll dig into the details of how all these members of the NPF are specified in practice, and what these terms really mean. For now, we simply note the range of options and tradeoffs. To recap, we've (schematically!) thought about how to parameterise a map from observed context sets $\mathcal{C}$ to predictive distributions at any target inputs $\mathbf{x}_{\mathcal{T}}$. Next, we consider how to _train_ such a map, i.e. how to learn the parameters $\theta$.

## Meta-Training in the NPF

In order to perform meta-learning, we are going to need a meta-dataset or a _dataset of datasets_. In the meta-learning literature, each dataset in the meta-dataset is referred to as a _task_. For the NPF, this means having access to many independent samples of functions from the data-generating process. Each sampled function is then a task. For example, we may have a large collection of audio waveforms $\{ \mathcal{D}_i \}_{i=1}^{N_{\mathrm{tasks}}}$ of people speaking. Each of these waveforms is itself a time-series $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$, where each $x^{(n)}, y^{(n)}$ is a timestamp/audio amplitude pair. Or we might have a large collection of natural images. Then each $\mathcal{D}$ would be a task consisting of many pixel-location/pixel-value pairs.

We would like to use this meta-dataset to learn how to make predictions at a target set upon observing a context set. To do this, we use an _episodic training procedure_, common in meta-learning, for the NPF. Each episode can be summarised in five steps:

1. Sample a task $\mathcal{D}$ from $\{ \mathcal{D}_i \}_{i=1}^{N_{\mathrm{tasks}}}$.
2. Randomly split the task into context and target sets: $\mathcal{D} = \mathcal{C} \cup \mathcal{T}$.
3. Pass $\mathcal{C}$ through the Neural Process to obtain the predictive distribution at the target inputs, $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$.
4. Compute an objective function $\mathcal{L}$ which measures the predictive performance on the target set.[^objective]
5. Compute the gradient $\nabla_{\theta}\mathcal{L}$ for stochastic gradient optimisation.

The episodes are repeated until training converges. How should we choose $\mathcal{L}$? The simplest choice is to let it be the log-likelihood of the target set:

$$
\begin{align}
\mathcal{L}(\theta ;\mathcal{C}, \mathcal{T}) = \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}).
\end{align}
$$

Other objectives have also been proposed, which will be discussed later. Intuitively, this procedure encourages the NPF to produce predictions that fit an unseen target set, given access to only the context set. Once meta-training is complete, if the Neural Process generalises well, it will be able to do this for brand new, unseen context sets. To recap, we've seen how the NPF can be thought of as a family of meta-learning algorithms, taking entire datasets as input, and providing predictions with a single forward pass.

```{admonition} Advanced$\qquad$Maximum-Likelihood Training
---
class: attention, dropdown
---

To better understand the objective $\mathcal{L}(\mathcal{C}, \mathcal{T} ; \theta) = \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ (which we refer to as _maximum-likelihood_ training), let $p(\mathcal{D}) = p(\mathcal{C}, \mathcal{T})$ be the _task distribution_, that is, the distribution from which we sample tasks in the episodic training procedure. Then, the true meta-objective can be written as an expectation over the task distribution:

$$
\begin{align}
\mathbb{E}_{p(\mathcal{D})} [\mathcal{L}(\theta ;\mathcal{C}, \mathcal{T})] &= \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}} , \mathbf{y}_{\mathcal{T}} )} [\log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})] \\
&= \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}})} \left[  \mathbb{E}_{p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})}  [ \log p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) ]  \right] \\
&= - \mathbb{E}_{p(\mathcal{C} , \mathbf{x}_{\mathcal{T}})} \left[ \mathrm{KL} (p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C}) \| p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})  ) \right] + \mathrm{const.}
\end{align}
$$

Here $\mathrm{KL}$ is the KL-divergence, which is a measure of how different two distributions are, and $\mathrm{const.}$ is a constant that is independent of $\theta$. Hence we can see that maximising the meta-learning objective is equivalent to _minimising the task averaged KL-divergence_ between the NPF predictive $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})$ and the conditional distribution $p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}} , \mathcal{C})$ of the task distribution.

This point of view also helps us see the potential issues with maximum-likelihood training if $p(\mathcal{D})$ is not truly representative of the data-generating process. For example, if all of the tasks sampled from $p(\mathcal{D})$ have their $x$-values bounded in a range $[-a, a]$, then the expected KL will not include any contributions from $p_\theta(y| x, \mathcal{C})$ when $x$ is outside of that range. Thus there is no direct incentive for the algorithm to learn a reasonable predictive distribution for that value of $x$. To avoid this problem, we should make our meta-dataset as representative of the tasks we expect to encounter in the future as possible.

Another way of addressing some of these issues is to bake appropriate _inductive biases_ into the Neural Process. In the next page of this tutorial, we will see that Convolutional Neural Processes use _translation equivariance_ to make accurate predictions for $x$-values outside the training range when the underlying data-generating process is stationary.


```

<!-- ## Stochastic Process Prediction

Let's look back to {numref}`ConvLNP`. Intuitively speaking, we can think of the context points as being observations of an _unknown, random function_. The solid blue lines of the NP are then samples from the predictive distribution of that function _conditioned on_ the observed context points. In mathematical terms, a probability distribution over a random function is called a _stochastic process_. This is where the name Neural Process comes from! The NPF is a method of using neural networks to specify a distribution over a random function.

How does this relate to the description we gave earlier? In the previous section we thought of the NPF as returning a predictive distribution $p_\theta(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ given a context set $\mathcal{C}$. However, in this expression, we are free to choose $\mathbf{x}_{\mathcal{T}}$ to be any target inputs we like. If we then choose the target inputs to be _the entire real line_, we are effectively specifying a distribution over $y(x)$ _for all_ $x$ --- in other words, a random function, i.e. a stochastic process. Hence the NPF can be thought of as learning a map from observed context sets to predictive stochastic processes.

This point of view of the NPF can provide good intuition and is helpful for making theoretical statements about the NPF. It also helps us contrast the Neural Process Family with another classical machine learning method for stochastic process prediction, _Gaussian processes_.

There are, however, several technical conditions that need to be satisfied to ensure that the NPF specifies a genuinely consistent stochastic process. These conditions are prominent in the original literature on Neural Processes, and help motivate some of the design choices in the NPF, such as the factorisation assumption in the CNPF. We provide a discussion of some of these issues in the dropdown box below. However, it is possible to follow the rest of this tutorial without covering this more technical section, so **feel free to skim or skip it on a first reading**. In summary, for a fixed context set $\mathcal{C}$, the NPF predictive distribution indeed defines a single consistent predictive stochastic process, although this may not be the case for varying context sets. -->

## Use Cases

What tasks can a trained Neural Process be applied to? Broadly speaking, they can be used in any situation where predictions need to be made under uncertainty. We list a few examples here.

* **Regression with uncertainty**. Imagine you're writing an algorithm to predict rainfall in the future, or to forecast demand for materials in your supply chain. These are all regression tasks where a well-trained Neural Process could exploit intricate structure in the data to make predictions and provide crucial uncertainty estimates.

* **Interpolating missing values**. This is closely related to the previous use case. Given a corrupted image or audio signal, we may want to see what the plausible interpolations are. Moreover, if the corrupted region is large, we want to be aware of the whole range of plausible interpolations, not just one guess. If the image is, say, a medical scan, this could be crucial for decision-making.

* **Active learning**. In active learning, our goal is to provide accurate predictions with as few measurements as possible. This is typically done by performing measurements at points with the greatest uncertainty. A Neural Process can be used to provide these uncertainty estimates. Once the measurement is taken, the new context point can be fed back into the Neural Process, and the uncertainty estimates can be updated for the next measurement.

* **Bayesian optimisation**. In Bayesian optimisation, the goal is to perform black-box optimisation of an unknown function when gradient information is unavailable, and each evaluation of the objective function is expensive. Most algorithms for Bayesian optimisation rely on querying points which have high expected reward and also high uncertainty, thus trading off exploration and exploitation.

## Summary of NPF Properties

Would an NPF be a good fit for your machine learning problem? To summarise, we note the advantages and disadvantages of the NPF:

* &#10003; **Fast predictions** on new context sets at test time. Often, training a machine learning model on a new dataset is computationally expensive. However, meta-learning allows the NPF to incorporate information from a new context set and make predictions with a _single_ forward pass. Typically the complexity will be linear or quadratic in the context set size instead of cubic as with standard Gaussian process regression.
* &#10003; **Well calibrated uncertainty**. Often meta-learning is applied to situations where each task has only a small number of examples at test time (also known as _few-shot learning_). These are exactly the situations where we should have uncertainty in our predictions, since there are many possible ways to interpolate a context set with few points. The NPF _learns_ to represent this uncertainty during episodic training.
* &#10003; **Data-driven expressivity**. The enormous flexibility of deep learning architectures means that the NPF can learn to model very intricate predictive distributions directly from the data. The user mainly has to specify the inductive biases of the network architecture, e.g. convolutional vs attentive.

However, these advantages come at the cost of the following disadvantages:

* &#10007; **The need for a large dataset for meta-training**. Meta-learning requires training on a large dataset of target and context points sampled from different functions, i.e., a large dataset of datasets. In some situations, a dataset of datasets may simply not be available. Furthermore, although predicting on a new context set after meta-training is fast, meta-training itself can be computationally expensive depending on the size of the network and the meta-dataset.

* &#10007; **Underfitting and smoothness issues**. The NPF predictive distribution has been known to underfit the context set, and also sometimes to provide unusually jagged predictions for regression tasks. The sharpness and diversity of the image samples for the LNPF could also be improved. However, improvements are being made on this front, with both the attentive and convolutional variants of the NPF providing significant advances.

In summary, we've taken a bird's eye view of the Neural Process Family and seen how they specify a map from datasets to stochastic processes, and how this map can be trained via meta-learning. We've also seen some of their use-cases and properties. Let's now dive into the actual architectures! In the next two pages we'll cover everything you need to know to get started with the models in Conditional and Neural Process Families.


[^objective]: Some Neural Processes also define the loss function to include the loss on the context set as well as the target set.

[^CNP]: {cite}`garnelo2018conditional`.

[^LNP]: {cite}`garnelo2018neural` --- in this paper and elsewhere in the Neural Process literature, the authors refer to latent neural processes simply as neural processes. In this tutorial we use the term 'neural process' to refer to both conditional neural processes and latent neural processes. We reserve the term 'latent neural process' specifically for the case when there is a stochastic latent variable $\mathbf{z}$.

[^AttnCNP]: {cite}`kim2019attentive` --- this paper only introduced the latent variable Attentive LNP, but one can easily drop the latent variable to obtain the Attentive CNP.

[^AttnLNP]: {cite}`kim2019attentive`.

[^ConvCNP]: {cite}`gordon2019convolutional`.

[^ConvLNP]: {cite}`foong2020convnp`.
