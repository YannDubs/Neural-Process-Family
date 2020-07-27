# Neural Process Family

% I don't know how to put them side by side :/

```{figure} ../gifs/ConvLNP_norbf_gp_extrap.gif
---
width: 35em
name: ConvLNP_norbf_gp_extrap
alt: Samples from ConvLNP trained on GPs
---
Sampled functions from ConvLNPs (Blue) and the oracle GP (Green) with Periodic or Noisy Matern kernel.
```

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_intro
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```



**Neural Processes Family (NPFs)** are a family of models that can be thought of as:
1. (Statistics Perspective) Modelling **stochastic processes** with neural networks by using large datasets instead of explicitely defining the finite-dimensional distribution.
2. (Deep Learning Perspective) Performing **meta learning** with uncertainty estimates.

```{figure} ../gifs/NPFs.gif
---
width: 30em
name: NPFs
alt: Schematic representation of CNPF computations
---
Schematic representation of NPF computations from [Marta Garnelo](https://www.martagarnelo.com/conditional-neural-processes).
```

{numref}`NPFs` [... high level intuition ? ...]

## Meta Learning Under Uncertainty

[...keep?...]

## Modeling Stochastic Processes

If you think of neural networks as a way of approximating a function $f : \mathcal{X} \to \mathcal{Y}$, then you can think of NPFs as a way of modeling a *distribution* over functions $f \sim P_{\mathcal{F}}$ (a stochastic process) conditioned on a certain set of points.

Specifically, we want to model a distribution over **target** values $\mathbf{y}_{\mathcal{T}} := \{y^{(t)}\}_{t=1}^T$ conditioned on a set of corresponding target features $\mathbf{x}_{\mathcal{T}} := \{x^{(t)}\}_{t=1}^T$ and a **context** set of feature-value pairs $\mathcal{C} := \{(x^{(c)}, y^{(c)})\}_{c=1}^C$.
We call such distribution $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$ the **posterior predictive** as it predicts $\mathbf{y}_{\mathcal{T}}$ after (a *posteriori*) conditioning on $\mathcal{C}$.

```{note}
If you sampled $\mathbf{y}_{\mathcal{T}}$ according to $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$, for all possible features $\mathbf{x}_{\mathcal{T}}=\mathcal{X}$ then you effectively sampled an entire function.
```

Outside of NPFs, stochastic processes are typically modelled by choosing the form of the distribution $p(\mathbf{y}|\mathbf{x})$ and then using rules of probabilities to infer different terms such as the posterior predictive $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \frac{p(\mathbf{y}_{\mathcal{T}},\mathbf{y}_{\mathcal{C}}|\mathbf{x}_{\mathcal{T}}, \mathbf{x}_\mathcal{C})}{p(\mathbf{y}_\mathcal{C}|\mathbf{x}_\mathcal{C})}$.
For example, this is the approach of [Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) (GPs), which uses a multivariate normal distribution for any $p(\mathbf{y}|\mathbf{x})$.
To define a proper stochastic process using $p(\mathbf{y}|\mathbf{x})$, it essentially needs to satisfy two consistency conditions ([Kolmogorov existence theorem](https://en.wikipedia.org/wiki/Kolmogorov_extension_theorem)) that can informally be summarized as follows :

* **Permutation invariance** 
$p(\mathbf{y}|\mathbf{x})=p(\pi(\mathbf{y})|\pi(\mathbf{x}))$ for all permutations $\pi$ on $1, ..., |\mathbf{y}|$. In other words, $\mathbf{x}$ and $\mathbf{y}$ are sets and should thus be unordered.

* **Consistent under marginalizaion** 
$p(\mathbf{y}|\mathbf{x})= \int p(\mathbf{y}'|\mathbf{x}') p(\mathbf{y}|\mathbf{x},\mathbf{y}',\mathbf{x}') d\mathbf{y}'$, for all $\mathbf{x}'$. 

By ensuring that these two condition hold, one can use standard probability rules and inherits nice mathematical properties. 
Unfortunately stochastic processes are usually computationally inefficient.
For example predicting values of a target set using GPs takes time which is cubic in the context set $\mathcal{O}(|\mathcal{C}|^3)$.


```{figure} ../images/computational_graph_NPFs.svg
---
width: 300em
name: computational_graph_NPFs
alt: high level computational graph of NPF
---
High level computational graph of the Neural Process Family.
```

In contrast, the core idea of NPFs is to directly model the posterior predictive using neural networks $p( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) \approx q_{\boldsymbol \theta}(\mathbf{y}_{\mathcal{T}}  | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$. 
It does so by encoding all the context set $\mathcal{C}$ to a global representation $R$ and then decoding from it for each target point.
The core of NPFs is that the encoder is permutation invariant, i.e. treats $\mathcal{C}$ as a set, to satisfy the first consistency condition of stochastic processes.


```{warning}
Although all NPFs use permutation invariant encoders to satisfy the first consistency condition, they usually are not consistent under marginalization and thus aren't proper stochastic processes. 
As a result, they sacrifice many nice mathematical properties.
[... Practical issues ? ...]
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

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.