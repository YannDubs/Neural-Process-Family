# Neural Process Family

In this tutorial, we introduce the **Neural Process Family** (NPF). Neural Processes (NPs) are a broad range of deep learning methods that model **predictive stochastic processes** via **meta-learning**. They are particularly well-suited to tasks which require fast test-time inference and well-calibrated uncertainty on continuous input domains. The goal of this tutorial is to provide a gentle introduction to the NPF. Our approach is to walk through several prominent members of the NPF, highlighting their key advantages and drawbacks. By accompanying the exposition with code and examples, we hope to (i) make the design choices and tradeoffs associated with each model clear and intuitive, and (ii) demonstrate both the simplicity of the framework and its broad applicability.

% I don't know how to put them side by side :/

```{figure} ../gifs/ConvLNP_norbf_gp_extrap.gif
---
width: 35em
name: ConvLNP_norbf_gp_extrap
alt: Samples from ConvLNP trained on GPs
---
Sample functions from the predictive stochastic process of ConvLNPs (blue) and the oracle GP (green) with periodic and noisy Matern kernels.
```

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_intro
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```

To motivate the NPF, consider the following tasks:

- Predicting **time-series data**. We are given a dataset $D = \{(x_n, y_n)\}_{n=1}^N$, where $x$ are the inputs (time) and $y$ are the outputs. We may want to predict the distribution of a new output $p(y_*|x_*, D)$ at any test location $x_*$, or to sample an entire time-series conditioned on $D$. For example, these could be audio waveforms, and $y$ might be the sound wave amplitude. In a corrupted audio signal, we may only have $D$ measured at a sparse set of points on the $x$-axis. We then want to make predictions that interpolate and extrapolate missing values of $y$. In regions of the $x$-axis with little data, there could be many reasonable predictions --- hence we don't just want point estimates but also measures of **uncertainty**. Ideally, we could also sample the entire time-series, so we could listen to many plausible completions of the data.

- Interpolating **image data** with uncertainty estimates. For example, we may be given satellite images of a region, which may be obscured by cloud-cover, or a medical imaging scan which has some occluded regions. In both cases we might be interested not just in a single interpolation, but in the _entire probability distribution_ over them. Images can be viewed as real-valued _functions_ on the two-dimensional plane. Each input $x$ is a two-dimensional vector denoting pixel location in the image, and each $y$ is a real number representing pixel intensity (or a three-dimensional vector for RGB images).

Each member of the NPF is a method for tackling problems such as these. To do this, the NPF brings together two key ideas: **stochastic process prediction** and **meta-learning**.

## Stochastic Process Prediction

First of all, what is a stochastic process (SP)? Intuitively speaking, an SP is a probability distribution over a random _function_. Each sample from a SP is a function whose domain is the whole real line (as with time-series data) or the two-dimensional plane (as with image data). Just as VAEs or normalizing flows try to model the probability distribution over random finite-dimensional vectors, we can think of stochastic processes as the infinite-dimensional analogue for random functions. From a Bayesian point of view, an SP represents a _state of belief about an unknown function_.

In the examples given above, the underlying SP might be "the distribution over all possible natural audio signals" or "the distribution over all possible medical image scans". Observing a dataset $D = \{(x_n, y_n)\}_{n=1}^N$ is then equivalent to sampling a function from the SP and observing it at a finite set of inputs $\{x_n\}_{n=1}^N$. Once we observe $D$, our state of belief about the function changes --- we _condition_ on $D$ to obtain a new SP, called the _predictive stochastic process_. This procedure is shown in {numref}`ConvLNP_norbf_gp_extrap` --- as points are observed, the error bars (light blue region) and predictive samples (dark blue lines) of the Neural Process get updated.

```{admonition} Gaussian Processes
---
class: tip, dropdown
---
{numref}`ConvLNP_norbf_gp_extrap` also shows the predictive mean and error-bars of the ground truth _Gaussian process_ (GP) used to generate the data. GPs are another way to specify stochastic processes. Unlike the NPF, GPs require the user to specify a kernel function to model the data. GPs are attractive due to the fact that exact prediction in GPs can be done _in closed form_. However, this has computational complexity $O(N^3)$ in the dataset size, which limits the application of exact GPs to large datasets. For an accessible introduction to GPs, see INSERT REF.

In this tutorial, we mainly use GPs to specify simple synthetic stochastic processes for benchmarking. We then train NPs to perform GP prediction. Since we can obtain the GP predictive distribution in closed form, we can use GPs to test the efficacy of NPs. It is important to remember, however, that NPs can model a broader range of SPs than GPs can (e.g. image data), are more computationally efficient at test time, and can learn directly from data without the need to hand-specify a kernel.   
```

From this perspective, SP prediction can be thought of as a _map_ from datasets $D$ to SPs --- for every dataset we might observe, the predictor returns the corresponding predictive SP. The NPF can be viewed as _deploying and training neural networks to approximate maps of this form_. To make this concrete, let the input space be $\mathcal{X}$ (e.g. time) and the output space be $\mathcal{Y}$ (e.g. amplitude). Let $f: \mathcal{X} \to \mathcal{Y}$ be a random function (e.g. an audio recording of someone speaking). We say that $f \sim P_f$, where $P_f$ is a stochastic process. Now assume we observe a dataset $D_c = \{(x_c, y_c)\}_{c=1}^C$. $D_c$ is the dataset we wish to condition on, to obtain a predictive SP denoted $P_{f|D_c}$. In the NPF literature, $D_c$ is referred to as a _context set_. (A SCHEMATIC DIAGRAM MIGHT BE GOOD HERE)

#### Target-set Consistency

To specify $P_{f|D_c}$ might seem impractical at first, as $f$ is infinite-dimensional. However, in practice we only need to specify the collection of _finite marginals_ of $P_{f|D_c}$: the distributions of $\{f(x_t) \}_{t=1}^T$ for all possible finite collections of test inputs $(x_1, ..., x_T)$. Think of $(x_1, ..., x_T)$ as the locations where we wish to query the predictive SP. In the NPF literature, these are known as _target inputs_ or _target features_. The collection of target inputs with their corresponding outputs, $D_t = \{(x_t, y_t)\}_{t=1}^T$, is called a _target set_.  For a fixed choice of target inputs, $\{f(x_t)\}_{t=1}^T$ is just a finite-dimensional random vector. However, we cannot just choose any set of finite marginals to specify an SP: they must be _consistent_ with each other for varying target sets, or else we might give contradictory predictions depending on which target set we consider.

```{admonition} Examples of Inconsistency
---
class: tip, dropdown
---
To illustrate the importance of consistency in specifying an SP, let's look at some artificial examples of finite marginals that are _not_ consistent. Let $x_1, x_2$ be two target inputs.

1. Consider a collection of finite marginals with $f(x_1) \sim \mathcal{N}(0, 1)$ and $[f(x_1), f(x_2)] \sim \mathcal{N}([10, 0], \mathbf{I})$. What is the mean of $f(x_1)$?
2. Consider a collection with $[f(x_1), f(x_2)] \sim \mathcal{N}([0, 0], \mathbf{I})$ and $[f(x_2), f(x_1)] \sim \mathcal{N}([1, 1], \mathbf{I})$. What is the mean of $f(x_1)$? What is the mean of $f(x_2)$?

We see the problem here: inconsistent marginals lead to self-contradictory predictions! In the first example, the marginals were not _consistent under marginalisation_: marginalising out $f(x_2)$ from the distribution of $[f(x_1), f(x_2)]$ did not yield the distribution of $f(x_1)$. In the second case, the marginals were not _consistent under permutation_: the distributions differed depending on whether you considered $f(x_1)$ first or $f(x_2)$ first.
```

The good news is that a well-known result called the Kolmogorov extension theorem states that as long as all the marginals satisfy these simple consistency requirements, they specify a well-defined SP. We'll refer to this kind of consistency as _target-set consistency_, since it relates to what happens when you vary the target set. Later, we'll look at various ways to use deep neural networks to specify SPs that are guaranteed to be target-set consistent.

#### Context-set Consistency

There is another kind of consistency that SP prediction must satisfy to obey the rules of probability theory, which involves more than just varying target sets. Consider two input-output pairs, $(x_1, y_1)$ and $(x_2, y_2)$. The product rule tells us that the joint predictive density must satisfy

\begin{align}
p(y_1, y_2| x_1, x_2) = p(y_1| x_1) p(y_2| y_1, x_1, x_2) = p(y_2| x_2) p(y_1| y_2, x_1, x_2) \label{AR}\tag{1}
\end{align}

We can see that this condition relates the finite marginals of SPs with varying context sets, and hence we refer to this as _context-set consistency_. If we computed our predictive SPs in exact form using the rules of probability theory, we would be guaranteed to make context-set consistent predictions. However, neural networks are general function approximators, and there's no simple way to constrain them to ensure that a member of the NPF is context-set consistent. In practice, the NPF yields good predictive performance even though it violates context-set consistency. This is because a well-trained Neural Process _learns_ a map that is approximately context-set consistent, even though it is not explicitly constrained to.

```{admonition} Relation to AR Models
---
class: tip, dropdown
---
Equation \ref{AR} may seem familiar --- this method of factorising a joint distribution into conditional ones is exactly that used in _autoregressive density models_. Such autoregressive (AR) models have been applied with great success for modelling a wide range of data, such as image data. However, to define a consistent density, AR models are usually defined with a _fixed_ ordering for the factorisation (e.g. left-to-right and top-to-bottom for the pixels in an image).

By contrast, the NPF allows us to evaluate the density of any target set given any context set. Hence the NPF may end up violating equation \ref{AR} if the relevant target and context sets are fed into it. One thing this means in practice is that if we were to use a Neural Process to sample functions autoregressively, we might get different distributions depending on what order we sampled the points.

```

```{admonition} Online Data
---
class: tip, dropdown
---
One situation where context-set consistency arises naturally is in _online data acquisition_. In this setting, we continually observe data and our context set grows with time. For example, we might first want to make a prediction for $y_1, y_2$ with an empty context set --- $p(y_1, y_2| x_1, x_2)$. Then we might later observe $y_1$. Equation \ref{AR} can then be seen as demanding that the updated prediction for $y_2$ conditioned on $y_1$ is consistent with the initial guess for $y_1, y_2$. This notion can be straightforwardly generalised to more than 2 datapoints.

```

To recap, we've seen that many tasks involving prediction under uncertainty can be framed as stochastic process prediction. And we've also seen that stochastic process prediction can be thought of as a map from context sets to stochastic processes. We've looked at how we can specify stochastic processes by defining their finite marginal distributions in a target-set consistent way. The NPF is a way of using deep neural networks to do exactly that. Next we'll look at how we might _train_ these neural networks via meta-learning.

## Meta-learning Under Uncertainty

What is meta-learning? Simply put, meta-learning is _learning to learn_. Whereas a standard machine learning algorithm is trained (e.g. by stochastic gradient descent) on a single dataset, and makes predictions at a new test point, a meta-learning algorithm is trained to _directly_ model the map from datasets to predictive distributions. In the case of the NPF, after meta-training is done, when presented with a new context set, a Neural Process does not need to perform an expensive gradient descent procedure to learn from the new context set. Instead, it receives the context set as input, and returns the predictive SP conditioned on that context set _in a single forward pass_.

In order to do this, each member of the NPF has to answer these questions: 1) how should we parameterise a map from datasets to predictive SPs? 2) how are we going to train this map?

#### Parameterising the Predictor

Let's first consider how to parameterise a map from datasets to stochastic processes using deep neural networks. The first challenge we have to address is how to input a _dataset_ into a neural network. This differs from standard vector-valued inputs in two ways:

* Datasets may have _varying sizes_: we might condition on a context set of size 1, 2 or 100. We would like the same architecture to be able to handle all these cases.
* Datasets have no intrinsic ordering. Hence any map $g$ acting on a dataset $E$ should be _permutation invariant_: $E(D) = E(\pi D)$, where $\pi$ is any permutation operator.

```{admonition} Deep Sets
---
class: tip, dropdown
---
In {cite}`zaheer2017deep`, the authors show that (subject to certain conditions) any permutation invariant map $E$ from a sequence $( x_n )_{n=1}^N$ to a real number can be expressed as:

$$
\begin{align}
E \left(( x_n )_{n=1}^N \right) = \rho \left ( \sum_{n=1}^N \phi(x_n) \right),
\end{align}
$$

for some suitable functions $\rho$ and $\phi$. This is known as a 'sum decomposition' or a 'Deep Sets encoding'. The NPF makes heavy use of sum-decompositions in its architectures. This result tells us that, as long as $\rho$ and $\phi$ are universal function approximators, this sum-decomposition can be done without loss of generality in terms of the class of permutation-invariant maps that can be expressed. We will later encounter a generalisation of this result for _translation equivariant_ functions, proven in {cite}`gordon2019convolutional`.

```

To tackle these requirements, members of the NPF first encode each point in the context set $D_c$ separately: $(x^{(c)}, y^{(c)}) \mapsto R^{(c)}$ for each $(x^{(c)}, y^{(c)} \in D_c$. These can be thought of as "local encodings" --- one for each context point. These local encodings are then aggregated into a single global encoding $R$ using an aggregation function $\mathrm{Agg}$. Importantly, $\mathrm{Agg}$ is _permutation invariant_: for any permutation $\pi$ of $\{ 1, 2, ..., C \}$,

$$
\mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)=\mathrm{Agg}\left(\pi\left(\{R^{(c)}\}_{c=1}^{C} \right)\right)
$$

The global encoding $R = \mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)$ can be thought of as a representation of the entire context set. This encoding may then undergo further processing by a deep neural network. At this point, the NPF splits into two sub-families, the _conditional_ Neural Process family (CNPF), and the _latent_ Neural Process family (LNPF):

* In the CNPF, the predictive distribution at any set of target inputs $\{ x^{(t)} \}_{t=1}^T$ is defined to be _factorised_ conditioned on $R$: $p(\mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \prod_{t=1}^T p(y^{(t)} | x^{(t)}, \mathcal{C})$

* In the LNPF, 

#### Meta-Training in the NPF

In order to perform meta-learning, we are going to need a meta-dataset or a _dataset of datasets_. In the meta-learning literature, each dataset in the meta-dataset is referred to as a _task_. For the NPF, this means having access to many observed samples of functions from the data-generating process $P$. Each sampled function is then a task. For example, we may have a large collection of audio waveforms $\{ D_i \}_{i=1}^{N_{\mathrm{tasks}}}$ of people speaking. Each of these waveforms is itself a time-series $D = \{ (x_j, y_j) \}_{j=1}^N$, where each $(x_j, y_j)$ is a timestamp/audio amplitude pair. Or we might have a large collection of natural images. Then each $D_i$ would be a task, consisting of many pixel-location/pixel-value pairs.

We next define an _episodic training procedure_ for the NPF. Each episode can be summarised in five steps:

1. Sample a task $D$ from $\{ D_i \}_{i=1}^{N_{\mathrm{tasks}}}$.
2. Randomly split the task into context and target sets: $D = D_c \cup D_t$.
3. Pass $D_c$ through the Neural Process to obtain a predictive stochastic process $P_{f|D_c}$.
4. Compute the objective function, measuring predictive performance on the target set $\mathcal{L}(D_t, P_{f|D_c})$.[^objective]
5. Compute the gradient of $\mathcal{L}$ with respect to the Neural Process parameters for stochastic gradient optimisation.

The episodes are then repeated until training converges. How should we choose $\mathcal{L}$? The simplest choice is to let it be the log-likelihood of the target set:

$$
\begin{align}
\mathcal{L}(D_t, P_{f|D_c}) = \log p( \{ y_t \}_{t=1}^T | \{ x_t \}_{t=1}^T, D_c),
\end{align}
$$

where $\log p( \{ y_t \}_{t=1}^T | \{ x_t \}_{t=1}^T, D_c)$ is obtained by evaluating the predictive SP $P_{f|D_c}$ at the target inputs $\{ x_t \}_{t=1}^T$. Intuitively, this procedure encourages the NPF to produce predictions that fit an unseen target set, given access to only the context set. Once meta-training is complete, if the Neural Process generalises well, it will be able to do this for brand new, unseen context sets.   

#### When is Meta-Learning Useful?


#### Relation to MAML

#### Relation to Probabilistic Meta-Learning

1. **Meta-learning**: The NPF is naturally geared for the meta-learning setting, also known as _learning to learn_. Arguably, the most prominent applications of meta-learning arise when the data for each task is sparse, motivating the need to account for **uncertainty**. As we shall see, NPF models provide an elegant framework for modelling uncertainty

```{figure} ../gifs/NPFs.gif
---
width: 30em
name: NPFs
alt: Schematic representation of CNPF computations
---
Schematic representation of NPF computations from [Marta Garnelo](https://www.martagarnelo.com/conditional-neural-processes).
```

## NPF Members

The major difference across the NPF, lies in the chosen encoder.
Usually (see {numref}`NPFs`) it first encode each context points separately ("local encoding") $x^{(c)}, y^{(c)} \mapsto R^{(c)}$ and then aggregates those in a representation for the entire context set $\mathrm{Agg}\left(\{R^{(c)}\}_{c=1}^{C} \right) \mapsto R$ .
Conditioned on this representation, all the target values become independent (factorization assumption) $q_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, R) =  \prod_{t=1}^{T} q_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, R)$.

NPFs can essentially be categorized into 2 sub-families depending on whether the global representation is treated as a latent variable ({doc}`Latent NPFs <LNPFs>`[^LNPs]) or not ({doc}`Conditional NPFs <CNPFs>`).
Inside each subfamily, the member essentially differ in how the representation is generated.
As we have already seen, the encoder is always permutation invariant. Specifically the aggregator $\mathrm{Agg}$ can be arbitrarily complex but is invariant to all permutations $\pi$ on $1, ..., C$:

```{math}
---
label: permut_inv
---
\begin{align}
\mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)=\mathrm{Agg}\left(\pi\left(\{R^{(c)}\}_{c=1}^{C} \right)\right)
\end{align}
```

```{admonition} Theory
---
class: dropdown, caution
---
[Talk about deep sets]
```

Here's an overly simplified summary of the different NPFs that we will consider in this directory:

[... keep? ...]

```{list-table} Summary of different NPFs
---
header-rows: 1
stub-columns: 1
name: summary_npf
---

* - Model
  - Aggregator
  - Stationarity
  - Expressivity
  - Extrapolation
  - Comp. Complexity
* - {doc}`CNP <../reproducibility/CNP>` {cite}`garnelo2018conditional`,{doc}`LNP <../reproducibility/LNP>` {cite}`garnelo2018neural`
  - mean
  -
  -
  -
  - $\mathcal{O}(T+C)$
* - {doc}`AttnCNP <../reproducibility/AttnCNP>` [^AttnCNP], {doc}`AttnLNP <../reproducibility/AttnLNP>` {cite}`kim2019attentive`
  - attention
  -
  - True
  -
  - $\mathcal{O}(C(T+C))$
* - {doc}`ConvCNP <../reproducibility/ConvCNP>` {cite}`gordon2019convolutional`,{doc}`ConvLNP <../reproducibility/ConvLNP>`  {cite}`foong2020convnp`.
  - convolution
  - True
  - True
  - True
  - $\mathcal{O}(U(T+C))$
```

## Training

[...MLE + meta data...]

The parameters are estimated by minimizing the expectation of the negative conditional conditional log likelihood (or approximation thereof for LNPFs):

```{math}
:label: training
\mathrm{NL}\mathcal{L}(\boldsymbol{\theta}) := -\mathbb{E}_{\mathrm{X}_\mathcal{T}} \left[ \mathbb{E}_{\mathrm{Y}_\mathcal{T}} \left[ \mathbb{E}_{\mathcal{C}} \left[ \log q_{\boldsymbol\theta} \left(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C} \right)\right] \right]\right]
```

We optimize it using stochastic gradient descent:

1. Sample size of context set $C \sim \mathrm{Unif}(0,|\mathcal{X}|)$
2. Sample $C$ context features $\mathbf{x}_{\mathcal{C}} \sim p(\mathbf{x}_{\mathcal{C}})$.
3. Sample associated values $\mathbf{y}_{\mathcal{C}} \sim p(\mathbf{y}_{\mathcal{C}} | \mathbf{x}_{\mathcal{C}})$.
4. Do the the same for the target set $\mathbf{x}_{\mathcal{T}},\mathbf{y}_{\mathcal{T}}$ [^sampleTargets].
5. Compute the MC gradients of the negative log likelihood $- \nabla_{\pmb \theta} \sum_{t=1}^{T} \log p_{\boldsymbol \theta}(y^{(t)} | \mathbf{y}_\mathcal{C}; \mathbf{x}_\mathcal{C}, x^{(t)})$
6. Backpropagate


In practice, it means that training NPFs requires a dataset over sets of points.
This contrasts with usual neural network training which requires only a dataset of points.
In deep learning terms, we would say that it is trained through a *meta-learning* procedure.

## Usecases

[...to do...]




## Properties

[...keep?...]

NPFs generally have the following desirable properties :

* &#10003; **Preserve permutation invariance** as with stochastic processes. This comes from the permutation invariance of $\mathrm{Agg}$ (Eq. {eq}`permut_inv`) and the factorization assumption (Eq. {eq}`formal`).
 for all permutations $\pi_{\mathcal{T}}: |\mathbf{y}_{\mathcal{T}}| \to |\mathbf{y}_{\mathcal{T}}|$ and $\pi_{\mathcal{C}}: |\mathbf{y}| \to |\mathbf{y}|$:

$$
p_{\boldsymbol \theta}(\mathbf{y}_{\mathcal{T}}  | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) = p_{\boldsymbol \theta}(\pi_{\mathcal{T}}(\mathbf{y})|\pi_{\mathcal{T}}(\mathbf{x}), \pi_{\mathcal{C}}(\mathcal{C}))
$$

* &#10003; **Data Driven Expressivity**. NPs require specification of prior knowledge through neural network architectures rather than a kernel function (like in GPs).
The former is usually less restrictive due to its large amount of parameters and removing the need of satisfying certain mathematical properties.
Intuitively, the NPs learn an "implicit kernel function" from the data.

* &#10003; **Test-Time Scalability**. Although the computational complexity depends on the NPF they are usually more computationally efficient (at test time) than proper stochastic processes.
Typically they will be linear or quadratic in the context set instead of cubic as with GPs.
[intuituin for gain????]

These advantages come at the cost of the following disadvantages

* &#10007; **Lack of consistency under marginalizaion**. So NPFs are not proper stochastic processes.
This essentially means that even if you had infinite computational power (to be able to marginalize) and sampled points autoregressively, the order in which you do it would change the distribution over targets.
Formally, there exists $\mathbf{x}'$ such that:

$$
p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C}) \neq \int p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C},\mathbf{y}',\mathbf{x}')  
p(\mathbf{y}'|\mathbf{x}', \mathcal{C}) d\mathbf{y}'
$$

%This lack of nice mathematical properties enable NPFs to predict in a more computationally efficient way.

* &#10007; **The need for large data**. Learning requires collecting and training on a large dataset of target and context points sampled from different functions. I.e. a large dataset of datasets.

* &#10007; $\sim$ **Lack of smoothness**. Due to highly non linear behaviour of neural networks and the factorized form of the predictive distribution (Eq. {eq}`formal`), the output tends to be non-smooth compared to a GP.
This is less true in newer NPFs such as ConvCNP {cite}`gordon2019convolutional`.





[^sampleTargets]: Instead of sampling targets, we will often use all the available targets $\mathbf{x}_\mathcal{T} = \mathcal{X}$. For example, in images the targets will usually be all pixels.

[^objective]: Some Neural Processes also define the loss function to include the loss on the context set as well as the target set.

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.
