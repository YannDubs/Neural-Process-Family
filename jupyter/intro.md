# Intro 

The aim of this section is to :
* Overview of Neural Process Family
* Overview of the framework I will use to try unifying different models from the neural process family

## Modeling Stochastic Processes

**Neural Processes Family (NPFs)** are a family of models which aim to model stochastic processes (distributions over functions) using neural networks.
If you think of neural networks as a way of approximating a function $f : \mathcal{X} \to \mathcal{Y}$, then you can think of NPFs as a way of modeling a distribution over functions conditioned on a certain set of points.

Specifically, we want to model a distribution over **target** values $\mathbf{y}_{\mathcal{T}} := \{y^{(t)}\}_{t=1}^T$ conditioned on a set of corresponding target features $\mathbf{x}_{\mathcal{T}} := \{x^{(t)}\}_{t=1}^T$ and a **context** set of feature-value pairs $\mathcal{C} := \{(x^{(c)}, y^{(c)})\}_{c=1}^C$.
We call such distribution $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$ the **posterior predictive**.

```{note}
If you sampled $\mathbf{y}_{\mathcal{T}}$ according to $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$, for all possible features $\mathbf{x}_{\mathcal{T}}=\mathcal{X}$ then you effectively sampled an entire function.
```

Prior to NPFs, one typically modelled stochastic processes by choosing the form of the distribution $p(\mathbf{y}|\mathbf{x})$ and then using rules of probabilities to infer different terms such as the posterior predictive $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C}) = \frac{p(\mathbf{y}_{\mathcal{T}},\mathbf{y}_{\mathcal{C}}|\mathbf{x}_{\mathcal{T}}, \mathbf{x}_\mathcal{C})}{p(\mathbf{y}_\mathcal{C}|\mathbf{x}_\mathcal{C})}$.
For example, this is the approach of [Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) (GPs), which uses a multivariate normal distribution for any $p(\mathbf{y}|\mathbf{x})$.
To define a proper stochastic process using $p(\mathbf{y}|\mathbf{x})$, it essentially needs to satisfy two consistency conditions ([Kolmogorov existence theorem](https://en.wikipedia.org/wiki/Kolmogorov_extension_theorem)) that can informally be summarized as follows :

* **Permutation invariance** 
$p(\mathbf{y}|\mathbf{x})=p(\pi(\mathbf{y})|\pi(\mathbf{x}))$ for all permutations $\pi$ on $1, ..., |\mathbf{y}|$. $\mathbf{x}$ and $\mathbf{y}$ are sets and should thus be unordered.

* **Consistent under marginalizaion** 
$p(\mathbf{y}|\mathbf{x})= \int p(\mathbf{y}'|\mathbf{x}') p(\mathbf{y}|\mathbf{x},\mathbf{y}',\mathbf{x}') d\mathbf{y}'$, for all $\mathbf{x}'$. 
$\mathbf{x}$ and $\mathbf{y}$ are sets and should thus be unordered.

By ensuring that these two condition hold, one can use standard probability rules and inherits nice mathematical properties. 
Unfortunately stochastic processes are usually computationally inefficient.
For example predicting values of a target set using GPs takes time which is cubic in the context set $\mathcal{O}(|\mathcal{C}|^3)$.
In contrast, the core idea of NPFs is to model directly the posterior predictive using neural networks $p( \mathbf{y}_{\mathcal{T}} | \mathbf{x}_{\mathcal{T}}, \mathcal{C}) \approx q_{\boldsymbol \theta}(\mathbf{y}_{\mathcal{T}}  | \mathbf{x}_{\mathcal{T}}, \mathcal{C})$ in the following way (sacrificing nice mathematical properties of stochastic processes for computational gains):

```{figure} images/NPFs.gif
---
height: 250px
name: NPFs
---
Schematic representation of NPF computations from [Marta Garnelo](https://www.martagarnelo.com/conditional-neural-processes).
```
```{figure} images/computational_graph_NPFs.svg
---
height: 250px
name: computational_graph_NPFs
---
High level computational graph of the Neural Process Family.
```

```{math}
:label: formal
\begin{align}
p(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{c}) 
&\approx q_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C}) & \text{Parametrization}\\
&= q_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, R) & \text{Sufficiency}  \\
&= \prod_{t=1}^{T} q_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, R)  & \text{Factorization}\\
&= \prod_{t=1}^{T} \mathcal{N}\left( y^{(t)};  \mu^{(t)},
\sigma^{2(t)}
\right) & \text{Gaussian} 
\end{align}
```

Where:

$$
\begin{align}
R^{(c)} 
&:= e_{\boldsymbol\theta}(x^{(c)}, y^{(c)}) & \text{Encoding} \\
R 
&:= \mathrm{Agg}\left(\{R^{(c)}\}_{c=1}^{C} \right) & \text{Aggregation} \\ 
(\mu^{(t)},\sigma^{2(t)}) 
&:= d_{\boldsymbol\theta}(x^{(t)},R) & \text{Decoding}  
\end{align}
$$

The aggregator is the major difference across the NPFs. Intuitively it takes in the representation of each context points separately and it aggregates those in a representation for the entire context set, but this aggregation can be arbitrarily complex and can be deterministic or stochastic. Given this representation, all the target values become independent (factorization assumption).
Importantly, the aggregator $\mathrm{Agg}$ is always invariant to all permutations $\pi$ on $1, ..., C$: 

```{math}
:label: permut_inv
\begin{align}
\mathrm{Agg}\left( \{R^{(c)}\}_{c=1}^{C} \right)=\mathrm{Agg}\left(\pi\left(\{R^{(c)}\}_{c=1}^{C} \right)\right) 
\end{align}
```

## Training

The parameters are estimated by minimizing the expectation of the negative conditional conditional log likelihood (or approximation thereof):

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

## Properties

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


## Usecases

There are two major usecases that we will consider for the posterior predictive $p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$ :

* **Sampling from the posterior predictive** $\mathbf{y}_{\mathcal{T}} \sim p(\mathbf{y}_{\mathcal{T}}|\mathbf{x}_{\mathcal{T}}, \mathcal{C})$.
This can be used for conditional generation, for example in active learning or Bayesian optimization.

* **Arg maximizing the posterior predictive** $\mathbf{y}_\mathcal{T}^* = \arg \max_{\mathbf{y}_\mathcal{T}'} p( \mathbf{y}_\mathcal{T}' | \mathbf{y}_\mathcal{C}; \mathbf{x}_\mathcal{C}, \mathbf{x}_\mathcal{T})$.
This can be used for imputing missing values or meta-learning.


## Members of NPFs

NPFs can essentially be categorized into 2 sub-families depending on whether they aim at sampling (Conditional NPFs) or at arg maximizing (Latent NPFs) the posterior predictive.
LNPFs essentially have have a latent random variable, while CNPFs don't.
Inside of each sub-family the main differences concerns the choice of aggregator $\mathrm{Agg}$ and the form of $R$.

We will be talking about and experimenting with the following:

* {doc}`Conditional NPFs <CNPFs>`(CNPFs)

    * {doc}`Conditional Neural Process <CNP>`(CNP) {cite}`garnelo2018conditional`.
    * {doc}`Attentive CNP <AttnCNP>` (AttnCNP){cite}`kim2019attentive` [^AttnCNP]. 
    * {doc}`Convolutional CNP <ConvCNP>` (ConvCNP) {cite}`gordon2019convolutional`. 

* {doc}`Latent NPFs <LNPFs>` (LNPFs) [^LNPs] 

    * {doc}`Latent Neural Process <LNP>`(LNP) {cite}`garnelo2018neural`.
    * {doc}`Attentive LNP <AttnLNP>`(AttnLNP) {cite}`kim2019attentive`. 
    * {doc}`Convolutional LNP <ConvLNP>`(ConvLNP) {cite}`foong2020convnp`. 



[^sampleTargets]: Instead of sampling targets, we will often use all the available targets $\mathbf{x}_\mathcal{T} = \mathcal{X}$. For example, in images the targets will usually be all pixels.

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.

[^LNPs]: In the literature the latent neural processes are just called neural processes. I use "latent" to distinguish them with the neural process family as a whole.