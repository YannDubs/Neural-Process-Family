# Conditional NPFs

```{figure} ../images/graph_model_CNPF.svg
---
width: 200em
name: graph_model_CNPF
---
Graphical model for the Conditional NPFs.
```

## Overview
We previously saw that the aim of NPFs is to model the posterior predictive $p( \mathbf{y}_\mathcal{T}' | \mathbf{y}_\mathcal{C}; \mathbf{x}_\mathcal{C}, \mathbf{x}_\mathcal{T})$, before diving into how we do so, let's take a step back and think about **why** we want to do so.
The usecase we will first consider is when we want to arg maximize the posterior predictive $\mathbf{y}_\mathcal{T}^* = \arg \max_{\mathbf{y}_\mathcal{T}'} p( \mathbf{y}_\mathcal{T}' | \mathbf{y}_\mathcal{C}; \mathbf{x}_\mathcal{C}, \mathbf{x}_\mathcal{T})$ in a computational efficient manner.
This can be used for imputing missing values or meta-learning [...].
For example in GPs you would first have to infer the posterior predictive (computationally prohibitive) and then it is very simple to compute the arg maximum as the predictive is Gaussian so the mode is simply the mean of the Gaussian.

Conditional Neural Process Sub-Family (CNPFs) are model that enable efficiently finding $\mathbf{y}_\mathcal{T}^*$ through forward pass of the network. The idea is to encode the entire context set in a fixed representation $R$ and to have a simple distribution of the target set conditioned on $R$.
To decrease the computational complexity we will assume that the targets are independent conditioned on $R$. 
What is more we will assume that they follow a Gaussian distribution, which enables efficient arg maximization by considering the predictive mean.
They all have the same probabilistic graphical model ({numref}`graph_model_CNPF`), specifically they model the posterior predictive as follows:


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

And the aggregator is permutation invariant (Eq. {eq}`permut_inv`).

The training of CNPFs is straight forward and consists in optimizing a the likelihood (Eq. {eq}`training`).

## Conditional Neural Process (CNP)

```{figure} ../images/computational_graph_CNPs.svg
---
width: 300em
name: computational_graph_CNPs_text
alt: Computational graph CNP
---
Computational graph for CNP.
```

Conditional Neural Process {cite}`garnelo2018conditional`:
*  use mean for aggregation because it’s a simple permutation invariant function (sum is commutative) + say how relates to deep sets
* computational cost $\mathcal{O}(T+C)$ + computational graph {numref}`computational_graph_CNPs_text`

Let's see how it works on some data. 
We first consider a simple 1D setting with samples from a GP with RBF kernel (data details in {doc}`Datasets Notebook <../reproducibility/Datasets>`):

```{figure} ../gifs/CNP_rbf.gif
---
width: 35em
name: CNP_rbf_text
alt: CNP on GP with RBF kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

From {numref}`CNP_rbf_text` we see that CNP performs quite well in this setting.
That being said it somewhat underfitts as seen by the fact that it  doesn't go through all the context points even though there is no noise.


```{figure} ../gifs/CNP_periodic.gif
---
width: 35em
name: CNP_periodic_text
alt: CNP on GP with Periodic kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with Periodic kernel.
```

This underfitting becomes apparent when the kernel is more complex as seen when the CNP is trained on a periodic GP.


```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`CNP Notebook <../reproducibility/CNP>`
```

## Attentive Conditional Neural Process (AttnCNP)


```{figure} ../images/computational_graph_AttnCNPs.svg
---
width: 300em
name: computational_graph_AttnCNPs_text
alt: Computational graph of AttnCNP
---
Computational graph for AttnCNP.
```



The major insight behind AttnCNP{cite}`kim2019attentive`[^AttnCNP] is that CNP underfitts because it uses a target independent representation, i.e. all the points in the context set have "the same importance" in the global representation. 
They alleviate this problem by using a target specific representation $R^{(t)}$  ({numref}`computational_graph_AttnCNPs_text`).

```{admonition} Advanced
---
class: dropdown, caution
---
[...keep...] ?
An other perspective, which we will be useful later on, is that the representation of the context $R$ is actually a function instead of a vector.
This function will be queried at the target position $x^{(t)}$ to yield a target specific vector representation $R^{(t)}$.
```

AttnCNP computes that representation using attention mechanism ({cite}`bahdanau2014neural`), which intuitively queries the context points with a target point to get a target specific representation of the context set.
[... quick explanation of attention ...]


```{admonition} Advanced
---
class: dropdown, caution
---
Why attention satisfies deep sets
```

To illustrate how this alleviates underfitting, imagine that you are querying a target point at the position of a context point, then you will mostly attent to that specific context point and will predict exactly assuming that there is no noise.
Note that this comes at the cost of additional computational complexity: from  $\mathcal{O}(T+C)$ to $\mathcal{O}(C(C*T))$ although it can be efficiently parallelized on GPUs.

```{admonition} Computational Complexity
---
class: dropdown, important
---
For every target we now have to attend to a representaion at each context element (cross attention).
For each target point a representation $R^{(t)}$ is then computed using cross-attention $\mathcal{O}(C*T)$.
In addition, the context representations (target indepent) usually first go through a self attention layer, where each context point attends to one another $\mathcal{O}(C^2)$.
Putting all together that gives  $\mathcal{O}(C(C*T))$ computational complexity.
```

Without further due, let us see how well it perform in practice. We will first evaluate it on GPs from different kernels (RBF, periodic, Noisy Matern).

```{figure} ../gifs/AttnCNP_single_gp.gif
---
width: 35em
name: AttnCNP_single_gp_text
alt: AttnCNP on GPs with RBF, periodic, noisy Matern kernel
---
Posterior predictive of AttnCNPs (Blue) and the oracle GP (Green) with RBF,Periodic,noisy Matern kernel.
```

 {numref}`AttnCNP_single_gp_text` shows that AttnCNP does not really suffer from underfitting and perform much better than CNP on challenging kernels.
 That being said it is still not perfect:
 * Periodic kernel not great
 *  Looking carefully at the Matern and RBF kernel, we also see that AttnCNP has a posterior predictive with "kinks", i.e., it is not very smooth. Note that kinks usually appear in the middle of 2 context points, we believe that they are a consequence of AttnCNP the abruptly changing its attention from one context point to the other (due to the exponential in the softmax).

Overall, AttnCNP performs quite well in this simple setting. Let us investigate more realistic setting, when we do not have access to the underlying data generating process : images.
Note that each image can indeed be seen as a 2D function mapping pixel location to values, so we can try to model the underlying stochastic process. 
Just as in 1D we want to model the posterior distribution of target pixels (all the image) given a few context pixels.

```{figure} ../gifs/AttnCNP_img_interp.gif
---
width: 30em
name: AttnCNP_img_interp_text
alt: AttnCNP on CelebA and MNIST
---
Posterior predictive of an AttnCNP for CelebA $32\times32$ and MNIST.
```

Again we see from  {numref}`AttnCNP_img_interp_text` that AttnCNP perform well.
This really shows an advantage of CNPs compared to other ways of modelling stochastic processes, i.e., it can model complicated underlying stochastic processes from data instead of having to select a kernel.
Intuitively it implicitely models a kernel using the data and through the architecture of the model.

One big problems with neural networks is that they are terrible at generalizing outside of the training distribution.
For example let us take an AttnCNP trained on MNIST (augmented with translations) and test it on a larger canvas with 2 digits (Zero Shot Multi MNIST, ZSMM).


```{figure} ../gifs/AttnCNP_img_extrap.gif
---
width: 15em
name: AttnCNP_img_extrap_text
alt: AttnCNP on ZSMM
---
Posterior predictive of an AttnCNP for ZSMM.
```

We see in {numref}`AttnCNP_rbf_extrap_text` that the model indeed completely breaks in this generalization task.
This is probably not surprising to anyone who worked with neural nets as the test set is significantly different of the training set, which is challenging.
But this issue arises in more subtle/benign. 
For example, let us query the CNP on points outside of the training interval (extrapolation) in the simple case of modelling a GP with RBF kernel


```{figure} ../gifs/AttnCNP_rbf_extrap.gif
---
width: 35em
name: AttnCNP_rbf_extrap_text
alt: extrapolation of AttnCNP on GPs with RBF kernel
---
Extrapolation (red dashes) of posterior predictive of AttnCNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`AttnCNP_rbf_extrap_text` clearly shows that AttnCNP breaks as soon as the target set is outside of the training regime, even tough the context points are also in the extrapolation regime. 
In other words, the AttnCNP really was not able to model the the fact that RBF kernel is stationary, i.e., that the absolute position of target points is not important only their relative position compared to context points.

```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`AttnCNP Notebook <../reproducibility/AttnCNP>`
```

## Convolutional Conditional Neural Process (ConvCNP)

```{figure} ../images/computational_graph_ConvCNPs.svg
---
height: 300em
name: computational_graph_ConvCNPs_text
alt: Computational graph ConvCNP
---
Computational graph of ConvCNP [to keep ? to simplify ?].
```

The idea[^disclaimer] behind ConvCNP {cite}`gordon2019convolutional` , is to bake in stationarity in the CNP to enable generalization outside of the training regime and sample efficiency.
One of the most famous inductive bias baked in in neural nets is the idea of translation equivariance through convolutions [cite].
It turns out that stationarity is equivalent to translation equivariant and ConvCNP takes advantage of this by using an encoder which is both permutation invariant and translation equivariant.
It does so by using convolutions in the encoder, but instead of the usual convolutions it has to define convolutions in sets (to keep permutation invariance) [...]
Functional representation [...], discretization [...], {numref}`computational_graph_ConvCNPs_text` computational graph [to keep ? to simplify ?], Intuition of theorem 1 [...]

```{admonition} Advanced
---
class: dropdown, caution
---
ConvDeepSet representation theorem [...] 
More information in {doc}`AttnCNP Notebook <Theory>`
```

```{admonition} Computational Complexity
---
class: dropdown, important
---
$\mathcal{O}(U(C*T))$ [...].
```

Now that the model is translation equivariant, we can test it in the more challenging extrapolation regime.

```{figure} ../gifs/ConvCNP_single_gp_extrap.gif
---
width: 35em
name: ConvCNP_single_gp_extrap_text
---
Extrapolation (red dashes) of posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with RBF,periodic,Matern kernel.
```

{numref}`ConvCNP_single_gp_extrap_text` shows that ConvCNP performs very well:
* No underffiting
* Can extrapolate outside of the training range due to its translation equivariance. Note that there is no free lunch, this only happens because the underlying stochastic process is stationary.
* smooth  "no kinks".
* It perform well on the periodic kernel. 



````{warning} 
The periodic kernel example is a little misleading, indeed the ConvCNP does not recover the underlying GP.
Specifically, it cannot because it has a bounded receptive field and as a result can only model local periodicity as illusrated when considering a much larger target interval ($[-2,14]$ instead of $[0,4]$). [... to keep ...?]

```{figure} ../gifs/ConvCNP_periodic_large_extrap.gif
---
width: 35em
name: ConvCNP_periodic_large_extrap_text
alt: ConvCNP on single images
---
Large extrapolation (red dashes) of posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with periodic kernel.
```

````


Let us see whether it also performs well in the more challenging case of images.

```{figure} ../gifs/ConvCNP_img.gif
---
width: 45em
name: ConvCNP_img_text
alt: ConvCNP on CelebA, MNIST, ZSMM
---

Posterior predictive of an AttnCNP for CelebA, MNIST, and ZSMM.
```

From {numref}`ConvCNP_img_text` we see that ConvCNP performs quite well on all datasets when the context set is large enough and uniformly sampled, even when extrapolation is needed (ZSMM).
However, it does not perform great when the context set is very small or when it is structured, e.g., half images. Note that seems more of an issue for ConvCNP compared to AttnCNP ({numref}`AttnCNP_img`). We hypothesize that this happens because the effective receptive field of the former is too small.
This issue can be alleviated by reducing the size of the context set seen during training (to force the model to have a large receptive field).

Although the zero shot generalization when performing on ZSMM are encouraging, it still is a simple artificial dataset. 
Let us consider a more complex zero shot generalization, namely we will evaluate the large model trained on CelebA128 on a image with multiple faces of different scale and orientation.

```{figure} ../images/ConvCNP_img_zeroshot.png
---
width: 50em
name: ConvCNP_img_zeroshot_text
alt: Zero shot generalization of ConvCNP to a real picture
---
Zero shot generalization of a ConvCNP trained on CelebA and evaluated on Ellen's selfie
```

From {numref}`ConvCNP_img_zeroshot_text` we see that the model is able to reasonably well generalize to real world data in a zero shot fashion.

Although the previous results look nice the usecases are not obvious as it is not very common to have missing pixels.
One possible application, is increasing the resolution of an image by querying positions "in between" pixels [... same figure as in intro, should we just link/referenve to it instead ? ]:

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_text
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```

From {numref}`ConvCNP_superes_text` we see that NPFs can indeed be used to increase the resolution of an image, even though it was not trained to do so! Results can probably be improved by training NPFs in such setting.


```{admonition} Details
---
class: tip
---
Model details, training and many more plots in {doc}`ConvCNP Notebook <../reproducibility/ConvCNP>`
```

(issues-cnpfs)=
### Issues With CNPFs

Although the results look quite good there are still some issues with CNPFs.
The first issue is that it due to the factorized form of the predictive distribution (Eq. {eq}`formal`), CNPFs cannot be used to sample coherent functions from the posterior predictive as it assumes that all the target distributions are independent given the context set.
Sampling from the posterior correponds to adding independent noise to the mean at each target location:


```{figure} ../images/ConvCNP_rbf_samples.png
---
width: 35em
name: ConvCNP_rbf_samples_text
alt: Sampling from ConvCNP on GP with RBF kernel
---
Samples form the posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with RBF kernel.
```
 
An other issue of CNPFs is that they cannot model complex multi modal posterior distribution, as the posterior distribution at every target point is always a Gaussian [....]

```{figure} ../images/ConvCNP_marginal.png
---
width: 20em
name: ConvCNP_marginal_text
alt: Samples from ConvCNP on MNIST and posterior of different pixels
---
Samples form the posterior predictive of ConvCNPs on MNIST (left) and posterior predictive of some pixels (right).
```

We will see how to solve both these issues by treating the representation as a latent variable in LNPFs.

[^disclaimer]: Disclaimer : 2 co-authors

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.