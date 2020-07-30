# Conditional NPFs

## Overview

```{figure} ../images/graph_model_CNPF.svg
---
width: 200em
name: graph_model_CNPF
---
Graphical model for the Conditional NPFs.
```

As we have seen, the key design choice of members of the NPF is how to model $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{C}, \mathbf{x}_\mathcal{T})$.
One simplifying assumption that we could make, illustrated in {numref}`graph_model_CNPF`, is that the predictive distribution _factorises_ conditioned on the context set.
This means that, having observed the context set $\mathcal{C}$, the prediction for each target location is _independent_ of any other target locations.
We can concisely express this assumption as

```{math}
---
label: conditional_predictives
---
\begin{align}
p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C}) = \prod_{t=1}^{T} p \left( y^{(t)} | x^{(t)}, \mathcal{C} \right).
\end{align}
```

We collectively refer to members of the NPF that employ this simplifying assumption as _conditional_ NP models, and to this sub-family as the CNPF.
Now, recall that the central idea of the NPF is to _locally_ encode each input-output pair in $\mathcal{C}$, and then _aggregate_ the encodings into a single representation of the context set, which we denote $R$.
A typical (though not _necessary_) choice is to consider Gaussian distributions for these predictives.
Putting these together, we can express the predictive distribution of CNPFs as

```{math}
:label: formal
\begin{align}
p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, R) & \text{Parameterisation}  \\
&= \prod_{t=1}^{T} p_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, R)  & \text{Factorisation}\\
&= \prod_{t=1}^{T} \mathcal{N} \left( y^{(t)};  \mu^{(t)}, \sigma^{2(t)} \right) & \text{Gaussian}
\end{align}
```

Where:

$$
\begin{align}
R^{(c)}
&:= e_{\boldsymbol\theta} \left(x^{(c)}, y^{(c)} \right) & \text{Encoding} \\
R
&:= \mathrm{Agg}\left(\{R^{(c)}\}_{c=1}^{C} \right) & \text{Aggregation} \\
(\mu^{(t)},\sigma^{2(t)})
&:= d_{\boldsymbol\theta}(x^{(t)},R) & \text{Decoding}  
\end{align}
$$

CNPFs make an important tradeoff.
On one hand, we have placed a severe restriction on the class of models that we can fit, and this restriction has important consequences.
We discuss these consequences in further detail, as well as provide some illustrations, at the end of this chapter.
On the other hand, the factorisation assumption makes evaluation of the predictive likelihoods tractable.
This means that we can employ exact maximum-likelihood procedures to train the model parameters.

```{admonition} Advanced
---
class: tip, dropdown
---
In the {doc}`Theory <Theory>` chapter we discuss the implications of an unbiased training procedure, and formalise what such a training procedure converges to.
```

Members of CNPF are distinguished by how they parameterise each of the key components:
* the encoding function $e_{\boldsymbol\theta}$,
* the aggregation function $\mathrm{Agg}$, and
* the decoder $d_{\boldsymbol\theta}$.

Next, we detail some promininent members of the CNPF, and discuss some of their advantages and disadvantages.


## Conditional Neural Process (CNP)

Arguably the simplest member of the CNPF, and the first considered in the literature, is the Conditional Neural Process (CNP) {cite}`garnelo2018conditional`.
The CNP is defined by the following design choices:
* The encoding function is defined by a feedforward MLP that takes as input the concatenation of $x^{(c)}$ and $y^{(c)}$, and outputs a simple vector $R^{(c)} \in \mathbb{R}^{dr}$.
*  The aggregation function is a simple mean over the representations $\frac{1}{| \mathcal{C} |} \sum_{c=1}^{| \mathcal{C} |} R^{(c)}$.
* The decoder $d_{\boldsymbol\theta}$ is also defined by a feedforward MLP that takes in the concatenation of $R$ and $x^{t}$, and outputs a mean $\mu$ and standard deviation $\sigma$.


```{admonition} Advanced
---
class: dropdown, caution
---
You should convince yourself (if it is not immediately obvious) that averaging is indeed permutation invariant. Hint: the summation operation is commutative, and this is followed by simple division by $| \mathcal{C} |$ to achieve "averaging".
```
```{admonition} Advanced
---
class: dropdown, caution
---
There is a strong relationship between this form and DeepSets networks, introduced by {cite}`zaheer2017deep`.
In the {doc}`Theory <Theory>` chapter we leverage this relationship to prove that CNPs can recover maps to any (continous) mean and variance functions.
```

```{admonition} Implementation Detail
---
class: tip, dropdown
---
Typically, we parameterise $d_{\boldsymbol\theta}$ as outputting $(\mu, \log \sigma)$, i.e., the _log_ standard deviation, so as to ensure that no negative variances occur.
```

The resulting computational graph is illustrated in {numref}`computational_graph_CNPs_text`. It is easy to see that the computational cost of making predictions for $T$ target set points conditioned on $\mathcal{C}$ with this design is $\mathcal{O}(T+|\mathcal{C}|)$.


```{figure} ../images/computational_graph_CNPs.svg
---
width: 300em
name: computational_graph_CNPs_text
alt: Computational graph CNP
---
Computational graph for CNP.
```


Let's see what prediction with such a model looks like in practice.
We first consider a simple 1D setting with samples from a GP with a radial basis function (RBF) kernel (data details in {doc}`Datasets Notebook <../reproducibility/Datasets>`).
Throughout the tutorial, we refer to similar experiments (though we vary the kernel) quite often.
Besides providing useful (and aesthetically pleasing) visualisations, the GPs admit ground truth predictive distributions, which allow us to compare to the "best possible" distributions.

```{figure} ../gifs/CNP_rbf.gif
---
width: 35em
name: CNP_rbf_text
alt: CNP on GP with RBF kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`CNP_rbf_text` provides the predictive distribution for a CNP trained on many samples from such a GP.
The blue line (shaded region) is the predicted mean function (two standard deviations).
The green solid (dashed) line is the ground truth mean (two standard deviations) from the "oracle" GP.
The figure demonstrates that the CNP performs quite well in this setting.
As more data is observed, the predictions become tighter.
Moreover, we can see that the CNP predictions quite accurately track the ground truth predictive.

That being said, looking closely we can see some signs that resemble underfitting: for example, the predictive mean does not pass through all the context points, despite there being no noise in the data-generating distribution.
The underfitting becomes abundantly clear when considering more complicated kernels, such as a periodic kernel.


```{figure} ../gifs/CNP_periodic.gif
---
width: 35em
name: CNP_periodic_text
alt: CNP on GP with Periodic kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with Periodic kernel.
```

Here we see that the CNP completely fails to model the predictive distribution: the mean function is overly smooth, hardly passes through the context points, and no notion of periodicity seems to have been learned.
Moreover, the uncertainty seems constant, and is significantly overestimated everywhere.
One potential solution, motivated by the fact that we know CNPs should be able to approximate _any_ mean and variance functions, might be to simply increase capacity of networks $e_{\boldsymbol\theta}$ and $d_{\boldsymbol\theta}$.
Unfortunately, it turns out that CNPs' modelling power scales quite poorly with the capacity of it networks.
A more promising avenue, which we explore next, is to consider the inductive biases of its architectures.


```{admonition} Details
---
class: tip
---
Model details and (many more) plots, along with code for constructing and training, CNPs can be found in {doc}`CNP Notebook <../reproducibility/CNP>`
```

## Attentive Conditional Neural Process (AttnCNP)


An important observation, made by {cite}`kim2019attentive`, is that in the CNP parameterisation, all points in the target set share a single, global representation $R$.
In other words, the CNP employs the same, _target independent representation_ when making predictions for any target set location.
This implies that all points in the context set have the same "importance", regardless of the location at which a prediction is being made.
Their proposed solution to this problem is to use a target-specific representation $R^{(t)}$, as illustrated in {numref}`computational_graph_AttnCNPs_text`.


```{figure} ../images/computational_graph_AttnCNPs.svg
---
width: 300em
name: computational_graph_AttnCNPs_text
alt: Computational graph of AttnCNP
---
Computational graph for AttnCNP.
```
```{admonition} Advanced
---
class: dropdown, caution
---
An other perspective, which we will be useful later on, is that the representation of the context $R$ is actually a function of the form $R : \mathcal{X} \to \mathbb{R}^{dr}$ instead of a vector.
This function will be queried at the target position $x^{(t)}$ to yield a target specific vector representation $R^{(t)}$.
```

To achieve this, {cite}`kim2019attentive` propose the Attentive CNP (AttnCNP[^AttnCNP]), which employs an _attention mechanism_ ({cite}`bahdanau2014neural`) to compute $R^{(t)}$.
There are many great resources about the use of attention mechanisms in machine learning [(LINKS TO PAPERS / BLOG POSTS ON ATTENTION?], and we encourage readers unfamiliar with the concept to look through these.
For our purposes, it suffices to think of attention mechanisms as learning to _attend_ to specific parts of an input that are particularly relevant to the desired output, giving them more _weight_ than others when making a prediction.

```{admonition} Advanced
---
class: dropdown, caution
---
To see that attention mechanisms satisfy permutation invariance, we must provide a more explicit form for the resulting representations.
With an attention mechanism, we can express these forms as

$$
\begin{align}
R^{(t)} = \sum_{c=1}^{| \mathcal{C} |} w_{\boldsymbol \theta}(x^{(c)}, x^{(t)}) R^{(c)},
\end{align}
$$

where $w_{\boldsymbol \theta}(x^{(c)}, x^{(t)})$ is the weight output by the attention mechanism, and $R^{(c)} = e_{\boldsymbol \theta}(x^{(c)}, y^{(c)})$ as before.
Thus, we can see that the resulting representation is simply a weighted sum, which is permutation invariant due to the commutativity of the summation operation.

From this view, we can also immediately see that the AttnCNP is a strict generalisation of the CNP, in the sense that the AttnCNP recovers the CNP if we set $w_{\boldsymbol \theta}(x^{(c)}, x^{(t)}) = \frac{1}{| \mathcal{C} |}$, for any $x^{(t)}$.
```

In the AttnCNP, we can think of the attention mechanism as providing a set of weights for $\{ w^{(c)}(x^{(t)}) \}$, one for each point in the context set.
Importantly, this set of weights is different for (and depends directly on) every target location!
To illustrate how this alleviates underfitting, imagine that our context set contains to points which are "very far" apart in $X$-space.
When making predictions close to the first point, we should largely ignore the $R^{(2)}$, since it contains little information about this region of $X$-space compared to $R^{(1)}$.
The converse is true when making predictions near the second point.
Attention allows us to parameterise and generalise this intuition, and learn it directly from the data!

```{admonition} Computational Complexity
---
class: dropdown, important
---
Note that attention comes at the cost of additional computational complexity: from  $\mathcal{O}(T+C)$ to $\mathcal{O}(C(C*T))$ although it can be efficiently parallelised on GPUs.
For every target we now have to attend to a representation at each context element (cross attention).
For each target point a representation $R^{(t)}$ is then computed using cross-attention $\mathcal{O}(C*T)$.
In addition, the context representations (target independent) usually first go through a self attention layer, where each context point attends to one another $\mathcal{O}(C^2)$.
Putting all together that gives  $\mathcal{O}(C(C*T))$ computational complexity.
```

Without further ado, let us see how the AttnCNP performs in practice.
We will first evaluate it on GPs from different kernels (RBF, periodic, and Noisy Matern).

```{figure} ../gifs/AttnCNP_single_gp.gif
---
width: 35em
name: AttnCNP_single_gp_text
alt: AttnCNP on GPs with RBF, periodic, noisy Matern kernel
---
Posterior predictive of AttnCNPs (Blue) and the oracle GP (Green) with RBF,Periodic,noisy Matern kernel.
```

{numref}`AttnCNP_single_gp_text` demonstrates that, as desired, AttnCNP alleviates many of the underfitting issues of the CNP, and generally performs much better on the challenging kernels.
However, looking closely at the resulting fits, we can still see some dissatisfying properties:
* the fit on the Periodic kernel is still not _great_, and
* looking carefully, we see that the AttnCNP has a posterior predictive with "kinks", i.e., it is not very smooth. Note that kinks usually appear between 2 context points. This leads us to believe that they are a consequence of the AttnCNP abruptly changing its attention from one context point to the other (due to the exponential in the softmax used to parameterise the attention mechanism).

Overall, AttnCNP performs quite well in this setting.
Next, we turn our attention (pun intended) to a more realistic setting, where we do not have access to the underlying data generating process: images.
In our experiments, we consider images as functions from the 2d integer grid (denoted $\mathbb{Z}^2$) to pixel space (this can be grey-scale or RGB, depending on the context).
When thinking about images as functions, it makes sense to reference the underlying stochastic process that generated them, though of course we can not express this process, nor have access to its ground truth posterior predictive distributions.
Just as in the 1D case, our goal is to model the posterior distribution of target pixels (typically the complete image) given a few context pixels.

```{figure} ../gifs/AttnCNP_img_interp.gif
---
width: 30em
name: AttnCNP_img_interp_text
alt: AttnCNP on CelebA and MNIST
---
Posterior predictive of an AttnCNP for CelebA $32\times32$ and MNIST.
```

{numref}`AttnCNP_img_interp_text` illustrates the performance of the AttnCNP on image reconstruction tasks with CelebA (left) and MNIST (right).
The results are quite impressive, and we can see that the AttnCNP is able to learn complicated structure in the underlying process, and produce compelling reconstructions of the data when faced with small context sets, and also structured obfuscation of the image that is different from what was observed during training.
The image experiments hammer home an important advantage of the NPF over other approaches to modelling stochastic processes.
We can see that the same architecture can scale to complicated underlying processes, learning the important properties from the data (rather than requiring intricate kernel design, as in the case of GPs).
Intuitively, we can think of NPF models as implicitly learning these properties, or kernels, from the data, while allowing us to bake in some inductive biases into the architecture of the model.


## Generalisation and Extrapolation

So far, we have seen that CNPF members can flexibly model a range of stochastic processes, and that we can overcome some of the key limitations by carefully designing the model architectures.
This led us to the AttnCNP, which achieves compelling performance on a range of both GP and image-based tasks.
Next, we consider the question of _generalisation_ and _exptrapolation_ with CNPF members.

One property of GPs is that they can condition predictions on data observed in any region of $X$-space.
When training a NPF member, we must specify a bounded range of $X$ on which data is observed, since we can never sample data from an infinite range.
We know that neural networks are typically quite bad at generalising outside the training distribution, and so we might suspect that CNPF model will not have this appealing property.

Let's first probe this question in the GP experiments.
To do so, we can examine what happens when trained models are provided with observations from the underlying process, but outside the training range.

```{figure} ../gifs/AttnCNP_rbf_extrap.gif
---
width: 35em
name: AttnCNP_rbf_extrap_text
alt: extrapolation of AttnCNP on GPs with RBF kernel
---
Extrapolation (red dashes) of posterior predictive of AttnCNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`AttnCNP_rbf_extrap_text` clearly shows that the AttnCNP breaks as soon as the context set contains observations from outside the training range.
In other words, the AttnCNP really was not able to model the the fact that RBF kernel is stationary, i.e., that the absolute position of target points is not important only their relative position compared to context points.

We can also observe this phenomenon in the image setting.
For example let us take an AttnCNP trained on MNIST (augmented with translations) and test it on a larger canvas with 2 digits (Zero Shot Multi MNIST, ZSMM).


```{figure} ../gifs/AttnCNP_img_extrap.gif
---
width: 15em
name: AttnCNP_img_extrap_text
alt: AttnCNP on ZSMM
---
Posterior predictive of an AttnCNP for ZSMM.
```

Again we see in {numref}`AttnCNP_rbf_extrap_text` that the model completely breaks in this generalisation task.
This is probably not surprising to anyone who worked with neural nets as the test set is significantly different of the training set, which is challenging.
Despite the challenging nature of the failure mode, it turns out that we can in fact construct NPF members that avoid it.
This leads us to our next CNPF member -- the ConvCNP.


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
More information in {doc}`Additional Theory <Theory>`
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
