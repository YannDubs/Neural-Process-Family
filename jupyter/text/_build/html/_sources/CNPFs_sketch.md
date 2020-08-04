# Conditional NPFs

## Overview

```{figure} ../images/graph_model_CNPF.svg
---
width: 200em
name: graph_model_CNPF
---
Graphical model for the Conditional NPFs.
```

As we have seen, the key design choice of members of the NPF is how to model the predictive distribution $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$.
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

A typical (though not _necessary_) choice is to consider Gaussian distributions for these likelihoods.
We collectively refer to members of the NPF that employ the factorisation assumption as _conditional_ NP models, and to this sub-family as the CNPF.
Now, recall that one guiding principle of the NPF is to _locally_ encode each input-output pair in $\mathcal{C}$, and then _aggregate_ the encodings into a single representation of the context set, which we denote $R$.
Putting these together, we can express the predictive distribution of CNPF members as

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

CNPF members make an important tradeoff.
On one hand, we have placed a severe restriction on the class of models that we can fit, and this restriction has important consequences.
We discuss these consequences in further detail, as well as provide some illustrations, at the end of this chapter.
On the other hand, the factorisation assumption makes evaluation of the predictive likelihoods tractable.
This means that we can employ simple and exact maximum-likelihood procedures to train the model parameters.

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

Next, we detail some prominent members of the CNPF, and discuss some of their advantages and disadvantages.


## Conditional Neural Process (CNP)

Arguably the simplest member of the CNPF, and the first considered in the literature, is the Conditional Neural Process (CNP) {cite}`garnelo2018conditional`.
The CNP is defined by the following design choices:
* The encoding function is defined by a feedforward MLP that takes as input the concatenation of $x^{(c)}$ and $y^{(c)}$, and outputs a simple vector $R^{(c)} \in \mathbb{R}^{dr}$.
*  The aggregation function is a simple average over the representations $\frac{1}{| \mathcal{C} |} \sum_{c=1}^{| \mathcal{C} |} R^{(c)}$.
* The decoder $d_{\boldsymbol\theta}$ is also defined by a feedforward MLP that takes in the concatenation of $R$ and $x^{(t)}$, and outputs a mean $\mu^{(t)}$ and standard deviation $\sigma^{(t)}$.


```{admonition} Note
---
class: dropdown, tip
---
You should convince yourself (if it is not immediately obvious) that averaging is indeed permutation invariant. Hint: the summation operation is commutative, and this is followed by simple division by $| \mathcal{C} |$ to achieve "averaging".
```
```{admonition} Details
---
class: dropdown, tip
---
There is a strong relationship between this form and DeepSets networks, introduced by {cite}`zaheer2017deep`.
In the {doc}`Theory <Theory>` chapter we leverage this relationship to prove that CNPs can recover maps to any (continuous) mean and variance functions.
```

```{admonition} Implementation Detail
---
class: tip, dropdown
---
Typically, we parameterise $d_{\boldsymbol\theta}$ as outputting $(\mu^{(t)}, \log \sigma^{(t)})$, i.e., the _log_ standard deviation, so as to ensure that no negative variances occur.
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
Besides providing useful (and aesthetically pleasing) visualisations, the GPs admit ground truth predictive distributions, which allow us to compare to the "best possible" distributions for a given context set.
In particular, if the CNP was "perfect", it would exactly match the predictions of the oracle GP.

```{figure} ../gifs/CNP_rbf.gif
---
width: 35em
name: CNP_rbf_text
alt: CNP on GP with RBF kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

{numref}`CNP_rbf_text` provides the predictive distribution for a CNP trained on many samples from such a GP.
The blue line represents the predicted mean function, and the shaded region represents two standard deviations on each side.
Similarly, the green solid line represents the ground truth mean, while the green dashed lines represent two standard deviations for the "oracle" GP.
The figure demonstrates that the CNP performs quite well in this setting.
As more data is observed, the predictions become tighter, as we would hope.
Moreover, we can see that the CNP predictions quite accurately track the ground truth predictive distribution.

That being said, looking closely we can see some signs that resemble underfitting: for example, the predictive mean does not pass through all the context points, despite there being no noise in the data-generating distribution.
The underfitting becomes abundantly clear when considering more complicated kernels, such as a periodic kernel.
Samples from periodic GPs are random periodic functions.
One thing we can notice about the ground truth GP is that it when observing data in some region of $X$-space, it leverages the periodic structure, and  becomes confident about predictions at every period.


```{figure} ../gifs/CNP_periodic.gif
---
width: 35em
name: CNP_periodic_text
alt: CNP on GP with Periodic kernel
---
Posterior predictive of CNPs (Blue) and the oracle GP (Green) with Periodic kernel.
```

Here we see that the CNP completely fails to model the predictive distribution: the mean function is overly smooth and hardly passes through the context points.
Moreover, we might hope that a CNP trained on periodic samples might learn to leverage this structure, but here we see that no notion of periodicity seems to have been learned.
Moreover, the uncertainty seems constant, and is significantly overestimated everywhere.
It thus seems reasonable to conclude that the CNP is not expressive enough to accurately model this (more complicated) process.

One potential solution, motivated by the fact that we know CNPs should be able to approximate _any_ mean and variance functions, might be to simply increase capacity of networks $e_{\boldsymbol\theta}$ and $d_{\boldsymbol\theta}$.
Unfortunately, it turns out that CNPs' modelling power scales quite poorly with the capacity of its networks.
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
Intuitively, it may seem natural that different points in the context set may be more relevant to predictions in different regions of $X$-space.
{cite}`kim2019attentive` propose to address this issue by using a target-specific representation $R^{(t)}$, as illustrated in {numref}`computational_graph_AttnCNPs_text`.

```{figure} ../images/computational_graph_AttnCNPs.svg
---
width: 300em
name: computational_graph_AttnCNPs_text
alt: Computational graph of AttnCNP
---
Computational graph for AttnCNP.
```
```{admonition} Note
---
class: dropdown, tip
---
Another perspective, which we will be useful later on, is that the representation $R$ is actually a function of the form $R : \mathcal{X} \to \mathbb{R}^{dr}$ instead of a vector.
This function will be queried at the target position $x^{(t)}$ to yield a target specific vector representation $R^{(t)}$.
```

To achieve this, {cite}`kim2019attentive` propose the Attentive CNP (AttnCNP[^AttnCNP]), which employs an _attention mechanism_ ({cite}`bahdanau2014neural`) to compute $R^{(t)}$.
There are many great resources available about the use of attention mechanisms in machine learning [(LINKS TO PAPERS / BLOG POSTS ON ATTENTION?], and we encourage readers unfamiliar with the concept to look through these.
For our purposes, it suffices to think of attention mechanisms as learning to _attend_ to specific parts of an input that are particularly relevant to the desired output, giving them more _weight_ than others when making a prediction.

```{admonition} Note
---
class: dropdown, tip
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
* the fit on the Periodic kernel is still not _great_.
In particular, looking closely, we can see that the mean and variance functions of the AttnCNP often fail to track the oracle GP, as they do not always leverage the periodic structure in the data.
* Moreover, we see that the AttnCNP has a posterior predictive with "kinks", i.e., it is not very smooth. Note that kinks usually appear between 2 context points. This leads us to believe that they are a consequence of the AttnCNP abruptly changing its attention from one context point to the other (due to the exponential in the softmax used to parameterise the attention mechanism).

Overall, AttnCNP performs quite well in this setting.
Next, we turn our attention (pun intended) to a more realistic setting, where we do not have access to the underlying data generating process: images.
In our experiments, we consider images as functions from the 2d integer grid (denoted $\mathbb{Z}^2$) to pixel space (this can be grey-scale or RGB, depending on the context).

```{figure} ../images/images_as_functions.png
---
width: 30em
name: images_as_functions_text
---
Viewing images as functions from $\mathbb{Z}^2 \to \mathbb{R}$.
```

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
Extrapolation (red dashes) of posterior predictive of AttnCNPs (Blue) and the oracle GP (Green) with RBF kernel. Left of the red vertical line is the training range, everything to the right is the "extrapolation range".
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

```{admonition} Disclaimer
---
class: tip
---
The authors of this tutorial are co-authors on the ConvCNP paper.
```

It turns out that the question of generalisation is closely linked to notions of _symmetry_ and _equivariance_.
In particular, the type of generalisation we are looking for -- that the NPF members be able to condition predictions on data observed anywhere in $X$-space -- can be mathematically expressed as a property called _translation equivariance_.
In the {doc}`Theory <Theory>` chapter we provide formal definitions for these notions, but for now the following intuition for translation equivariance (TE) should suffice: if observations are shifted in time or space, then the resulting predictions should be shifted by the same amount.
It turns out that this simple inductive bias, when appropriate, is _wildly_ effective.
Arguably the most prominent example of translation equivariance in machine learning are convolutional neural networks (CNNs), which are built around this simple intuition.

This provides the central motivation behind the ConvCNP {cite}`gordon2019convolutional`: how can we bake TE into CNPF members, while preserving other desirable aspects of the models?
Hopefully, doing so should lead to CNPF members that exhibit the generalisation capacities we are after, as well as improved performance and _parameter efficiency_ (which is another benefit often associated with baking in TE).
To achieve this, {cite}`gordon2019convolutional` propose a special form of convolutional layer, which can be seen as an extension of standard convolutions to _set-structured_ inputs.
We refer to such layers as _SetConvs_.
With this layer, we can then construct ConvCNPs by defining the following design choices (illustrated in {numref}`computational_graph_ConvCNPs_text`):
* The encoding function is a SetConv layer.
* The aggregation function is a standard convolutional layer.
* The decoder $d_{\boldsymbol\theta}$ is a standard CNN.

```{admonition} Important
---
class: dropdown, caution
---
The ConvCNP encoder _explicitly_ encodes $\mathcal{C}$ into a space of functions.
That is, every context set is represented by a _continuous_ function $X \to \mathbb{R}^{dr}$.
While this representation is extremely useful, it requires an approximation: since the decoder is a CNN, it operates on _discrete_ inputs.
To enable this, the ConvCNP first _discretises_ the functional representation of the context set before passing it to the decoder.
While this means the forward pass through the ConvCNP can only be approximated, in practice this turns out to not have a detrimental effect on performance.
```

```{admonition} Advanced
---
class: dropdown, caution
---
One of the key results of {cite}`gordon2019convolutional` is to show that SetConv layers are both permutation invariant and translation equivariant.
Moreover, {cite}`gordon2019convolutional` demonstrate that _any_ (continuous) permutation invariance and TE function can be represented by a ConvCNP.
The proof relies on first extending the DeepSets work of {cite}`zaheer2017deep` to include TE as well.
In the {doc}`Theory<Theory>` chapter, we provide a sketch of this proof.
```

```{admonition} Computational Complexity
---
class: dropdown, important
---
$\mathcal{O}(U(C*T))$ [...].
```

```{figure} ../images/computational_graph_ConvCNPs.svg
---
height: 300em
name: computational_graph_ConvCNPs_text
alt: Computational graph ConvCNP
---
Computational graph of ConvCNP [to keep ? to simplify ?].
```

Now that we have constructed a translation equivariant CNPF, we can test it in the more challenging extrapolation regime.
We begin with the same set of GP experiments, but this time already including data observed from outside the original training range.

```{figure} ../gifs/ConvCNP_single_gp_extrap.gif
---
width: 35em
name: ConvCNP_single_gp_extrap_text
---
Extrapolation (red dashes) of posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with (top) RBF, (center) periodic, and (bottom) Noisy Matern kernel.
```

{numref}`ConvCNP_single_gp_extrap_text` demonstrates that the ConvCNP indeed performs very well!
In particular, we can see that:
* Like the use of attention, the TE inductive bias also helps the model avoid the tendency to underfit the data.
* Unlike the other members of the CNPF, the ConvCNP is able to extrapolate outside of the training range.
Note that this is a direct consequence of TE.
* Unlike attention, the ConvCNP produces smooth mean and variance functions, avoiding the "kinks" introduced by the AttnCNP.
* The ConvCNP is able to learn about the underlying structure in the periodic kernel.
We can see this by noting that it produces periodic predictions, even "far" away from the observed data.


````{warning}
---
class: dropdown, important
---
The periodic kernel example is a little misleading, indeed the ConvCNP does not recover the underlying GP.
In fact, we know that it cannot exactly recover the underlying process, because it has a _bounded receptive field_, and as a result can only model local periodicity.
This is best seen when considering a much larger target interval ($[-2,14]$ instead of $[0,4]$), as below.

```{figure} ../gifs/ConvCNP_periodic_large_extrap.gif
---
width: 35em
name: ConvCNP_periodic_large_extrap_text
alt: ConvCNP on single images
---
Large extrapolation (red dashes) of posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with periodic kernel.
```

In fact, this is true for any of the GPs above, all of which have infinite receptive fields, meaning that no model with a bounded field (such as the ConvCNP) can ever exactly recover them.
This discussion alludes to one of the key design choices of the ConvCNP, which is the size of its receptive field.
Note that, unlike standard CNNs, the resulting receptive field does not only depend on the CNN architecture, but also on the granularity of the discretisation employed on the functional representation.
Thus, when designing ConvCNP architectures, some consideration should be given to the interplay between the discretisation and the architecture of the decoder.
````


Let us also examine the performance of the ConvCNP on the more challenging image experiments.
As with the AttnCNP, we consider CelebA and MNIST reconstruction experiments, but also unclude the ZSMM experiments that evaluate the model's ability to generalise beyond the training data.

```{figure} ../gifs/ConvCNP_img.gif
---
width: 45em
name: ConvCNP_img_text
alt: ConvCNP on CelebA, MNIST, ZSMM
---

Posterior predictive of an AttnCNP for CelebA, MNIST, and ZSMM.
```

From {numref}`ConvCNP_img_text` we see that the ConvCNP performs quite well on all datasets when the context set is large enough and uniformly sampled, even when extrapolation is needed (ZSMM).
However, performance is less impressive when the context set is very small or when it is structured, e.g., half images.
In our experiments we find that this is more of an issue for the ConvCNP than the AttnCNP ({numref}`AttnCNP_img`).
We hypothesise that this happens because the effective receptive field of the former is too small, whilst the AttnCNP has an infinite receptive field, allowing it to make more use of observed pixels far from the target pixel.
This issue can be alleviated by reducing the size of the context set seen during training (to force the model to have a large receptive field), but this solution is somewhat dissatisfying in that it requires tinkering with the training procedure.

Although the zero shot generalisation when performing on ZSMM are encouraging, the task is somewhat artificial.
Let us consider more complex zero shot generalisation tasks.
First, we will evaluate a ConvCNP trained on CelebA128 on an image with multiple faces of different scale and orientation.

```{figure} ../images/ConvCNP_img_zeroshot.png
---
width: 50em
name: ConvCNP_img_zeroshot_text
alt: Zero shot generalization of ConvCNP to a real picture
---
Zero shot generalization of a ConvCNP trained on CelebA and evaluated on Ellen's selfie
```

From {numref}`ConvCNP_img_zeroshot_text` we see that the model is able to generalise reasonably well to real world data in a zero shot fashion.
Although the previous results look nice, the use-cases are not immediately obvious, as it is not very to have missing pixels.
One possible application is increasing the resolution of an image.
This can be achieved by querying positions "in between" pixels.

```{figure} ../images/ConvCNP_superes.png
---
width: 35em
name: ConvCNP_superes_text
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP.
```

{numref}`ConvCNP_superes_text` demonstrates such an application.
We see that CNPFs can indeed be used to increase the resolution of an image, even though it was not trained to do so!


```{admonition} Details
---
class: tip
---
Model details, training and many more plots are available in the {doc}`ConvCNP Notebook <../reproducibility/ConvCNP>`
```

(issues-cnpfs)=
### Issues With CNPFs

Let's take a step back.
So far, we have seen that we can use the factorisation assumption to construct simple members of the CNPF, perhaps the simplest of these being the CNP.
Our first observation was that while the CNP can track underlying processes, it tends to underfit when the processes are more complicated.
We saw that this tendency can be addressed by adding appropriate inductive biases into the paramterisation of the model.
Specifically, the AttnCNP significantly improves upon the CNP by adding an attention mechanism to generate target-specific representations of the context set.
We further saw that the AttnCNP also scales nicely to image-settings.
However, both the CNP and AttnCNP fail to make meaningful predictions when data is observed outside the training range (or when the test distribution is different from training, i.e., in the ZSMM example).
Finally, we saw how including TE as an inductive bias led to well-fitting functions that generalised elegantly to observations outside the training range.

Let us now consider more closely the implications of the factorisation assumption, along with the Gaussian form of predictive distributions.
One immediate consequence of using the Gaussian likelihood is that we cannot recover multi-modal distributions.
To see why this might be an issue, consider making predictions for the MNIST reconstruction experiments.

```{figure} ../images/ConvCNP_marginal.png
---
width: 20em
name: ConvCNP_marginal_text
alt: Samples from ConvCNP on MNIST and posterior of different pixels
---
Samples form the posterior predictive of ConvCNPs on MNIST (left) and posterior predictive of some pixels (right).
```

Looking at {numref}`ConvCNP_marginal_text`, we might expect that sampling from the predictive distribution of an unobserved pixels sometimes yield completely white values, and sometimes completely black, depending on whether the sample represents, for example, a 3 or a 5.
However, a Gaussian distribution, which is uni-modal (see {numref}`ConvCNP_marginal_text` right), cannot achieve this type of multi-modal behaviour.

```{admonition} Note
---
class: tip, dropdown
---
One solution to this particular problem might be to employ some other parametric distribution that enables multimodality, for example, a mixture of Gaussians.
While this may solve some issues, we can generalise this point to say that the CNPF requires specifying _some parametric form of distribution_.
Ideally, what we would like is some parametrisation of the NPF that enables us to recover _any_ form of marginal distribution.
```

The other major restriction is the factorisation assumption itself, which has several important implications.
First, it means that the model can not leverage any correlation structure that might exist in the predictive distribution over multiple target sets.
For example imagine that we are modelling samples from an underlying GP.
If the model is making predictions at two target locations that are "close" in $X$-space, it seems reasonable that whenever it predicts the first be "high", it predict something similar for the second, and vice versa.
Yet the factorisation assumption means that the model cannot learn this type of structure.
Another implication is that we can not produce _coherent_ samples from the predictive distribution.
In fact, sampling from the posterior corresponds to adding independent noise to the mean at each target location, resulting in samples that look nothing like the underlying process.


```{figure} ../images/ConvCNP_rbf_samples.png
---
width: 35em
name: ConvCNP_rbf_samples_text
alt: Sampling from ConvCNP on GP with RBF kernel
---
Samples form the posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with RBF kernel.
```

```{admonition} Note
---
class: tip, dropdown
---
This inability to sample from the predictive may inhibit the deployment of CNPF members from several application areas for which it might otherwise be potentially well-suited.
One such example is the use of Thompson sampling algorithms for e.g., contextual bandits or Bayesian optimisation, which require a model to produce samples.
```

In the next chapter, we will see one approach to solving both these issues by treating the representation as a latent variable in the latent Neural Process sub-family.

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.
