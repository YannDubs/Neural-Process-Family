# Conditional NPF

## Overview

```{figure} ../images/graph_model_CNPF.svg
---
width: 300em
name: graph_model_CNPF
---
Probabilistic graphical model for the Conditional NPF.
```

The key design choice for members of the NPF is how to model the predictive distribution $p( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$.
In particular, we require the predictive distributions to be consistent with each other for different $\mathbf{x}_\mathcal{T}$, as discussed in the {ref}`previous section <target_consistency>`.
One of the simplest ways to ensure this is to define a predictive distribution that is *factorised* conditioned on the context set (as illustrated in {numref}`graph_model_CNPF`).
This means that, conditioned on the context set $\mathcal{C}$, the prediction at each target location is _independent_ of the prediction at any other target locations.
We can concisely express this assumption as:

```{math}
---
label: conditional_predictives
---
\begin{align}
p_{\boldsymbol\theta}( \mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C}) = \prod_{t=1}^{T} p_{\boldsymbol\theta} \left( y^{(t)} | x^{(t)}, \mathcal{C} \right).
\end{align}
```

A typical (though not necessary) choice is to set each of the individual terms in the likelihood to be a Gaussian density.
We collectively refer to members of the NPF that employ this factorisation assumption as _conditional_ NP models, and to this sub-family as the CNPF.
Now, recall that one guiding principle of the NPF is to _locally_ encode each input-output pair in $\mathcal{C}$, and then _aggregate_ the encodings into a single representation of the context set, which we denote by $R$.
Putting these together, we can express the predictive distribution of CNPF members as

```{math}
:label: formal
\begin{align}
p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})
&= p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, R) & \text{Parameterisation}  \\
&= \prod_{t=1}^{T} p_{\boldsymbol\theta}(y^{(t)} |  x^{(t)}, R)  & \text{Factorisation}\\
&= \prod_{t=1}^{T} \mathcal{N} \left( y^{(t)};  \mu^{(t)}, \sigma^{2(t)} \right) & \text{Gaussianity}
\end{align}
```

Where:

$$
\begin{align}
R^{(c)}
&:= \mathrm{Enc}_{\boldsymbol\theta} \left(x^{(c)}, y^{(c)} \right) & \text{Encoding} \\
R
&:= \mathrm{Agg}\left(\{R^{(c)}\}_{c=1}^{C} \right) & \text{Aggregation} \\
(\mu^{(t)},\sigma^{2(t)})
&:= \mathrm{Dec}_{\boldsymbol\theta}(x^{(t)},R) & \text{Decoding}
\end{align}
$$

```{admonition} Advanced$\qquad$Factorisation $\implies$ consistency
---
class: hint, dropdown
---
We show that the CNPF predictive distribution specifies a consistent stochastic processes, given a fixed context set $\mathcal{C}$. Recall that the consistency we require means consistency under _marginalisation_ and _permutation_. To see that the CNPF predictive distribution satisfies consistency under permutation, let $\mathbf{x}_{\mathcal{T}} = \{ x^{(t)} \}_{t=1}^T$ be target inputs. Let $\pi$ be a permutation of $\{1, ..., T\}$. Then the predictive density is (suppressing the $\mathcal{C}$-dependence):

$$
\begin{align}
    p_\theta(y^{(1)}, ..., y^{(T)} | x^{(1)}, ..., x^{(T)}) &= \prod_{t=1}^{T} p_\theta( y^{(t)} | x^{(t)}) \\
    &= p_\theta(y^{(\pi(1))}, ..., y^{(\pi(T))} | x^{(\pi(1))}, ..., x^{(\pi(T))}),
\end{align}
$$

since multiplication is commutative. To show consistency under marginalisation, consider two target inputs, $x^{(1)}, x^{(2)}$. Then by marginalising out the second target output, we get:

$$
\begin{align}
    \int p_\theta(y^{(1)}, y^{(2)}| x^{(1)}, x^{(2)}, \mathcal{C}) \, \mathrm{d}y^{(2)} &= \int p_\theta(y^{(1)}| x^{(1)}, R)p_\theta(y^{(2)}| x^{(2)}, R) \, \mathrm{d}y^{(2)} \\
    &= p_\theta(y^{(1)}| x^{(1)}, R) \int p_\theta(y^{(2)}| x^{(2)}, R) \, \mathrm{d}y^{(2)}\\
    &= p_\theta(y^{(1)}| x^{(1)} \mathcal{C}).
\end{align}
$$

which shows that the predictive distribution obtained by querying the CNPF member at $x^{(1)}$ is the same as that obtained by querying it at $x^{(1)}, x^{(2)}$ and then marginalising out the second target point. Of course, the same idea works with collections of any size, and marginalising any subset of the variables.

```

CNPF members make an important tradeoff.
On one hand, these assumptions place a severe restriction on the class of predictive stochastic processes we can model.
As discussed at the end of this chapter, this has important consequences, such as the inability of the CNPF to produce coherent samples.
On the other hand, the factorisation assumption makes evaluation of the predictive likelihoods analytically tractable.
This means we can employ a simple maximum-likelihood procedure to train the model parameters, i.e., training amounts to directly maximising the log-likelihood $\log p_{\boldsymbol\theta}(\mathbf{y}_\mathcal{T} | \mathbf{x}_\mathcal{T}, \mathcal{C})$, as discussed on the previous page.

```{admonition} Advanced$\qquad$Unbiasedness
---
class: hint, dropdown
---
NOT SURE WHAT WE HAVE PLANNED FOR THIS BOX :)
```

Now that we've given an overview of the entire CNPF, we'll discuss three particular members: the Conditional Neural Process (CNP), Attentive Conditional Neural Process (AttnCNP), and the Convolutional Conditional Neural Process (ConvCNP). Each member of the CNPF can be broadly distinguished by how they parameterise each of the key components:
* the encoder $\mathrm{Enc}_{\boldsymbol\theta}$,
* the aggregator $\mathrm{Agg}$, and
* the decoder $\mathrm{Dec}_{\boldsymbol\theta}$.

We begin by describing the Conditional Neural Process, arguably the simplest member of the CNPF, and the first considered in the literature.

(cnp)=
## Conditional Neural Process (CNP)

The Conditional Neural Process (CNP) {cite}`garnelo2018conditional` is defined by the following design choices:
* The encoder is a feedforward multi-layer perceptron (MLP) that takes as input the concatenation $[x^{(c)}; y^{(c)}]$, and outputs a vector $R^{(c)} \in \mathbb{R}^{dr}$.
*  The aggregator is a simple average over each local representation $\frac{1}{C} \sum_{c=1}^{C} R^{(c)}$.
* The decoder $d_{\boldsymbol\theta}$ is an MLP that takes as input the concatenation of the global representation and the target input, $[R; x^{(t)}]$, and outputs a predictive mean $\mu^{(t)}$ and variance $\sigma^{2(t)}$.

```{admonition} Note$\qquad$Implementation Detail
---
class: note, dropdown
---
Typically, we parameterise $\mathrm{Dec}_{\boldsymbol\theta}$ as outputting $(\mu^{(t)}, \log \sigma^{(t)})$, i.e., the _log_ standard deviation, so as to ensure that no negative variances occur.
```

The resulting computational graph is illustrated in {numref}`computational_graph_CNPs_text`. Notice that the computational cost of making predictions for $T$ target points conditioned on $C$ context points with this design is $\mathcal{O}(T+C)$.
Indeed, each $(x,y)$ pair of the context set is encoded independently ($\mathcal{O}(C)$) and the representation $R$ can then be re-used for predicting at each target location ($\mathcal{O}(T)$).
This means that once trained, CNPs are much more efficient than GPs (which scale as $\mathcal{O}((C+T)^3)$).


```{figure} ../images/computational_graph_CNPs.svg
---
width: 400em
name: computational_graph_CNPs_text
alt: Computational graph CNP
---
Computational graph for CNPs.
```

```{admonition} Advanced$\qquad$Universality and DeepSets
---
class: dropdown, tip
---
There is a strong relationship between this encoder+aggregator and a class of models known as DeepSets networks introduced by Zaheer et al.{cite}`zaheer2017deep`.
In particular, Zaheer et al. show that (subject to certain conditions) _any_ continuous function operating on sets ($S$) can be expressed as:

$$
\begin{align}
  f(S) = \rho \left( \sum_{s \in S} \phi(s) \right),
\end{align}
$$

for appropriate functions $\rho$ and $\phi$.
This is known as a 'sum decomposition' or 'Deep Sets encoding'.
This result tells us that, as long as $\rho$ and $\phi$ are universal function approximators, this sum-decomposition can be done without loss of generality in terms of the class of permutation-invariant maps that can be expressed.

CNPs make heavy use this type of architecture.
To highlight the similarities, we can express each of the mean and standard deviation functions of the CNP as

$$
\begin{align}
  f_\mu \left( x^{(t)}; \mathcal{C} \right) &= \rho_\mu \left( \left[x^{(t)}; \frac{1}{|\mathcal{C}|} \sum_{(x, y) \in \mathcal{C}} \phi([x;y])\right] \right), \\
  f_\sigma \left( x^{(t)}; \mathcal{C} \right) &= \rho_\sigma \left( \left[x^{(t)}; \frac{1}{|\mathcal{C}|} \sum_{(x, y) \in \mathcal{C}} \phi([x;y])\right] \right)
\end{align}
$$

where $[\cdot ; \cdot ]$ is the concatenation operation.
In fact, the only difference between the CNP form and the DeepSets of Zaheer et al. is the concatenation operation in $\rho$, and the use of the mean for aggregation, rather than summation.
It is possible (though we do not demonstrate this here) to leverage this relationship to show that CNPs can recover any continuous mean and variance functions, which provides justification for the proposed architecture.

```

Let's see what prediction using a CNP looks like in practice.
We first consider a simple 1D regression task trained on samples from a GP with a radial basis function (RBF, also known as _exponentiated quadratic_) kernel (data details in {doc}`Datasets Notebook <../reproducibility/Datasets>`).
Throughout the tutorial, we refer to similar experiments (though we vary the kernel) quite often.
Besides providing useful (and aesthetically pleasing) visualisations, the GPs admit closed form posterior predictive distributions, which allow us to compare to the "best possible" distributions for a given context set.
In particular, if the CNP was "perfect", it would exactly match the predictions of the oracle GP. However, unlike the GP, the CNP would at best only be able to obtain the correct marginal predictions, since it cannot model dependencies. We will discuss this further at the end of this page.

```{figure} ../gifs/CNP_rbf.gif
---
width: 35em
name: CNP_rbf_text
alt: CNP on GP with RBF kernel
---
Predictive distribution of a CNP (the blue line represents the predicted mean function and the shaded region shows a standard deviation on each side $[\mu-\sigma,\mu+\sigma]$) and the oracle GP (green line represents the ground truth mean and the dashed line show a standard deviations on each side) with RBF kernel.
```

{numref}`CNP_rbf_text` provides the predictive distribution for a CNP trained on many samples from such a GP.
The figure demonstrates that the CNP performs quite well in this setting.
As more data is observed, the predictions become tighter, as we would hope.
Moreover, we can see that the CNP predictions quite accurately track the ground truth predictive distribution.

That being said, we can see some signs that resemble underfitting: for example, the predictive mean does not pass through all the context points, despite there being no noise in the data-generating distribution.
The underfitting becomes abundantly clear when considering more complicated kernels, such as a periodic kernel (i.e. GPs generating random periodic functions as seen in {ref}`Datasets Notebook <samples_from_gp>`).
One thing we can notice about the ground truth GP predictions is that it leverages the periodic structure in its predictions, and  becomes confident at every period upon observing data in one period.

```{figure} ../gifs/CNP_periodic.gif
---
width: 35em
name: CNP_periodic_text
alt: CNP on GP with Periodic kernel
---
Posterior predictive of a CNP (Blue line for the mean with shaded area for $[\mu-\sigma,\mu+\sigma]$) and the oracle GP (Green line for the mean with dotted lines for +/- standard deviation) with Periodic kernel.
```

In contrast, we see that the CNP completely fails to model the predictive distribution: the mean function is overly smooth and hardly passes through the context points.
Moreover, we might hope that a CNP trained on the sample functions from this GP might learn to leverage this periodic structure, but it seems that no notion of periodicity has been learned in the predictions.
Finally, the uncertainty seems constant, and is significantly overestimated everywhere.
It seems that the CNP is has failed to learn the more complex structure of the optimal predictive distribution for this kernel.

Let's now test the CNP on a more interesting task, one where we do _not_ have access to the ground truth predictive distribution: image completion. As described on the previous page, an image can be viewed as a function from the 2D plane to $\mathbb{R}$ (for monochrome images) or to $\mathbb{R}^3$ (for RGB images). During meta-training, we treat each image as a sampled function, and split the image into context and target pixels. At test time, we can feed in a new context set and query the CNP at all the pixel locations, to interpolate the missing values in the image. {numref}`CNP_img_interp_text` shows the results:

```{figure} ../gifs/CNP_img_interp.gif
---
width: 30em
name: CNP_img_interp_text
alt: AttnCNP on CelebA and MNIST
---
Posterior predictive of a CNP on CelebA $32\times32$ and MNIST.
```

These results are quite impressive, however there are still some signs of underfitting. In particular, the interpolations are not very sharp, and do not totally resemble the ground truth image even when there are many context points. Nevertheless, this experiment demonstrates the power of neural processes: they can be applied out-of-the-box to learn this complicated structure directly from data, something that would be very difficult with a Gaussian process.

One potential solution to the overfitting problem, motivated by the fact that we know CNPs should be able to approximate _any_ mean and variance functions, might be to simply increase the capacity of the networks $\mathrm{Enc}_{\boldsymbol\theta}$ and $\mathrm{Dec}_{\boldsymbol\theta}$.
Unfortunately, it turns out that the CNP's modelling power scales quite poorly with the capacity of its networks.
A more promising avenue, which we explore next, is to consider the _inductive biases_ of its architectures.


```{admonition} Note
---
class: note
---
Model details and (many more) plots, along with code for constructing and training CNPs, can be found in {doc}`CNP Notebook <../reproducibility/CNP>`
```

(attncnp)=
## Attentive Conditional Neural Process (AttnCNP)


One possible explanation for CNP's underfitting is that all points in the target set share a _single_ global representation $R$ of the context set, i.e., $R$ is *independent* of the location of the target input.
This implies that all points in the context set are given the same "importance", regardless of the location at which a prediction is being made.
For example, CNPs struggle to take advantage of the fact that if a target point is very close to a context point, they will often both have similar values. One possible solution is to use a _target-specific_ representation $R^{(t)}$, as illustrated in {numref}`computational_graph_AttnCNPs_text`.

```{figure} ../images/computational_graph_AttnCNPs.svg
---
width: 300em
name: computational_graph_AttnCNPs_text
alt: Computational graph of AttnCNP
---
Computational graph for AttnCNPs.
```

```{admonition} Note$\qquad$Functional Representation
---
class: dropdown, note
---
Another perspective, which we will be useful later on, is that the representation $R$ is actually a function of the form $R : \mathcal{X} \to \mathbb{R}^{dr}$ instead of a vector.
This function will be queried at the target position $x^{(t)}$ to yield a target specific vector representation $R^{(t)}$.
```

To achieve this, Kim et al. {cite}`kim2019attentive` propose the Attentive CNP (AttnCNP[^AttnCNP]), which aggregates all the local $R^{(c)}$ in a target dependent representation $R^{(t)}$ using an _attention mechanism_ {cite}`bahdanau2014neural` instead of CNPs' mean aggregation.
There are many great resources available about the use of attention mechanisms in machine learning (e.g. [Distill's interactive visualisation](https://distill.pub/2016/augmented-rnns/#attentional-interfaces), [Lil'Log](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html), or the [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)), and we encourage readers unfamiliar with the concept to look through these.
For our purposes, it suffices to think of attention mechanisms as learning to _attend_ to specific parts of an input that are particularly relevant to the desired output, giving them more _weight_ than others when making a prediction.

In the AttnCNP, we can think of the attention mechanism as a
set of weights $\{ w^{(c)}(x^{(t)}) \}$, one for each point in the context set.
Importantly, this set of weights is different for (and depends directly on) every target location!
In other words, the AttnCNP replaces the CNP's simple average by a (more general) weighted average which gives a larger weight to "important" context points.


```{admonition} Note$\qquad$Permutation Invariance
---
class: dropdown, note
---
To see that attention mechanisms satisfy permutation invariance, we must provide a more explicit form for the resulting representations.
With an attention mechanism, we can express the target dependent representation as

$$
\begin{align}
R^{(t)} = \frac{1}{\sum_{c=1}^{C} w_{\boldsymbol \theta}(x^{(c)}, x^{(t)})} \sum_{c=1}^{C} w_{\boldsymbol \theta}(x^{(c)}, x^{(t)}) R^{(c)},
\end{align}
$$

where $w_{\boldsymbol \theta}(x^{(c)}, x^{(t)})$ is the weight output by the attention mechanism, and $R^{(c)} = \mathrm{Enc}_{\boldsymbol \theta}(x^{(c)}, y^{(c)})$ as before.
Thus, we can see that the resulting representation is simply a weighted average, which is permutation invariant due to the commutativity of the summation operation.
```

To illustrate how attention can alleviate underfitting, imagine that our context set contains two observations with inputs $x^{(1)}, x^{(2)}$ that are "very far" apart in input space. These observations (input-output pairs) are then mapped by the encoder to the local representations $R^{(1)}, R^{(2)}$ respectively.
Intuitively, when making predictions close to $x^{(1)}$, we should focus on $R^{(1)}$ and ignore $R^{(2)}$, since $R^{(1)}$ contains much more information about this region of input space. An attention mechanism allows us to parameterise and generalise this intuition, and learn it directly from the data!

This gain in expressivity comes at the cost of increased computational complexity, from  $\mathcal{O}(T+C)$ to $\mathcal{O}(T*C)$, as a representation of the context set now needs to be computed for each target point.

```{admonition} Warning$\qquad$Self-Attention
---
class: dropdown, warning
---
In the above discussion we only consider an attention mechanism between the target and context set (referred to as cross-attention).
In addition, AttnCNP often first uses an attention mechanism between context points: _self-attention_.

Notice that this does not impact the permutation invariance of the predictions --- if the context set is permuted, so will the set of local representations. When cross attention is applied, the ordering of this set is irrelevant, hence the predictions are unaffected.

Using a self-attention layer will also increase the computational complexity to $\mathcal{O}(C(C+T))$ as each context point will first attend to every other context point in $\mathcal{O}(C^2)$ before the cross attention layer which is $\mathcal{O}(C*T)$.
```

To summarise, the AttnCNP is defined by the following design choices:
* The encoding function is either an MLP (as with the CNP), or a _self-attention_ layer operating on the concatenation of $x^{(c)}$ and $y^{(c)}$, outputting vectors $R^{(c)} \in \mathbb{R}^{dr}$.
*  The aggregation function is defined by a cross-attention mechanism, which we can express as

$$
\begin{align}
R^{(t)} = \sum_{c=1}^{C} \tilde{w}_{\boldsymbol \theta} \left( x^{(c)}, x^{(t)} \right) R^{(c)},
\end{align}
$$

where $\tilde{w}$ are the _normalised_ attention weights.
* The decoder $d_{\boldsymbol\theta}$ is a MLP that takes in the concatenation of the target-point specific representation $R^{(t)}$ and the target location $x^{(t)}$, and outputs a mean $\mu^{(t)}$ and variance $\sigma^{2(t)}$.


Without further ado, let's see how the AttnCNP performs in practice.
We will first evaluate it on GP regression with different kernels (RBF, periodic, and Noisy Matern).

```{figure} ../gifs/AttnCNP_single_gp.gif
---
width: 35em
name: AttnCNP_single_gp_text
alt: AttnCNP on GPs with RBF, Periodic, Noisy Matern kernel
---

Posterior predictive of AttnCNPs (Blue line for the mean with shaded area for $[\mu-\sigma,\mu+\sigma]$) and the oracle GP (Green line for the mean with dotted lines for +/- standard deviation) with RBF, Periodic, and Noisy Matern kernel.
```

{numref}`AttnCNP_single_gp_text` demonstrates that, as desired, AttnCNP alleviates many of the underfitting issues of the CNP, and generally performs much better on the challenging kernels.
However, looking closely at the resulting fits, we can still see some dissatisfying properties:
* The fit on the Periodic kernel is still not _great_.
In particular, we see that the mean and variance functions of the AttnCNP often fail to track the oracle GP, as they do not really leverage the periodic structure.
* Moreover, the posterior predictive of the AttnCNP has "kinks", i.e., it is not very smooth. Notice that these kinks usually appear between 2 context points. This leads us to believe that they are a consequence of the AttnCNP abruptly changing its attention from one context point to the other.

Overall, AttnCNP performs quite well in this setting.
Next, we turn our attention (pun intended) to the image setting:

```{figure} ../gifs/AttnCNP_img_interp.gif
---
width: 30em
name: AttnCNP_img_interp_text
alt: AttnCNP on CelebA and MNIST
---
Posterior predictive of an AttnCNP for CelebA $32\times32$ and MNIST.
```

{numref}`AttnCNP_img_interp_text` illustrates the performance of the AttnCNP on image reconstruction tasks with CelebA (left) and MNIST (right). Note that the reconstructions are sharper than those for the CNP. Interestingly, when only a vertical or horizontal slice is shown, the ANP seems to 'blur' out its reconstruction somewhat.


<!-- The results are quite impressive, as the AttnCNP is able to learn complicated structure in the underlying process, and even produce descent reconstructions of structured obfuscation of the image that is different from the random context pixels seen during training.
The image experiments hammer home an important advantage of the NPF over other approaches to modelling stochastic processes:
the same architecture can scale to complicated underlying processes, learning the important properties from the data (rather than requiring intricate kernel design, as in the case of GPs).
Intuitively, we can think of NPF models as implicitly learning these properties, or kernels, from the data, while allowing us to bake in some inductive biases into the architecture of the model. -->


```{admonition} Note
---
class: note
---
Model details, training and many more plots in {doc}`AttnCNP Notebook <../reproducibility/AttnCNP>`
```

(extrapolation)=
## Generalisation and Extrapolation

So far, we have seen that well designed CNPF members can flexibly model a range of stochastic processes by being trained from functions sampled from the desired stochastic process.
Next, we consider the question of _generalisation_ and _extrapolation_ with CNPF members.

Let's begin by discussing these properties in GPs. Many GPs used in practice have a property known as _stationarity_. Roughly speaking, this means that the GP gives the same predictions regardless of the absolute position of the context set in input space --- only relative position matters. One reason this is useful is that stationary GPs will make sensible predictions regardless of the range of the inputs you give it. For example, imagine performing time-series prediction. As time goes on, the input range of the data increases. Stationary GPs can handle this without any issues.

In contrast, one downside of the CNP and AttnCNP is that it learns predictions solely through the data that it is presented with during meta-training. If this data has a limited range in input space, then there is no reason to believe that the CNP or AttnCNP will be able to make sensible predictions when queried outside of this range. In fact, we know that neural networks are typically quite bad at generalising outside their training distribution (i.e., in the _out-of-distribution_ (OOD) regime).

<!-- One property of GPs is that they can condition predictions on data observed in any region of $X$-space.
A possible downside from the fact that neural processes learn how to model a stochastic process from a dataset, is that during training we must specify a bounded range of $X$ on which data is observed, since we can never sample data from an infinite range.
We know that neural networks are typically quite bad at generalising outside the training distribution, and so we might suspect that CNPF models will not exhibit this appealing property. -->

Let's first probe this question on the 1D regression experiments.
To do so, we examine what happens when trained models are conditioned on observations from the ground truth GP, but located _outside_ the training range in input space.


```{figure} ../gifs/CNP_AttnCNP_rbf_extrap.gif
:width: 35em
:name: CNP_AttnCNP_rbf_extrap_text
:alt: extrapolation of CNP on GPs with RBF kernel

Extrapolation of posterior predictive of CNP (Top) and AttnCNP (Bottom) and the oracle GP (Green) with RBF kernel. Left of the red vertical line is the training range, everything to the right is the "extrapolation range".
```

{numref}`CNP_AttnCNP_rbf_extrap_text` clearly shows that the CNP and the AttnCNP  break as soon as the context set contains observations from outside the training range.
In other words, they are not able to model the fact that the RBF kernel is stationary, i.e., that the absolute position of target points is not important, but only their relative position to the context points.
Interestingly, they both fail in different ways: the CNP seems to fail for any target location, while the AttnCNP fails only when the target locations are in the extrapolation regime (it still performs well in the training regime). This is perhaps because the AttnCNP predictions only attend to nearby context set points.

We can also observe this phenomenon in the image setting.
For example let us evaluate the CNP and AttnCNP on Zero Shot Multi MNIST (ZSMM) where the training set consists of MNIST examples (augmented with translations) while the test images are larger canvases with 2 digits (training and testing examples are shown in the {doc}`Datasets Notebook <../reproducibility/Datasets>`)).


% The following puts 2 figures side by side in large screens but one after the other on phones

````{panels}
:container: container-fluid
:column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-0 m-0
:card: shadow-none border-0

```{figure} ../gifs/CNP_img_extrap.gif
:width: 100%
:name: CNP_img_extrap_text
:alt: CNP on ZSMM

Posterior predictive of an CNP for ZSMM.
```

---

```{figure} ../gifs/AttnCNP_img_extrap.gif
:width: 100%
:name: AttnCNP_img_extrap_text
:alt: AttnCNP on ZSMM

Posterior predictive of an AttnCNP for ZSMM.
```

````

Again we see in {numref}`CNP_img_extrap_text` and {numref}`AttnCNP_img_extrap_text` that the models completely break in this generalisation task --- both models still expect only a single centered digit. They are unable to spatially extrapolate to multiple, uncentered digits.
This is likely not surprising to anyone who has worked with neural nets as the test set here is significantly different of the training set.
Despite the challenging nature of this task, it turns out that we can in fact construct NPF members that perform well, by building in the appropriate inductive biases.
This leads us to our next CNPF member -- the ConvCNP.


(convcnp)=
## Convolutional Conditional Neural Process (ConvCNP)

```{admonition} Disclaimer
---
class: attention
---
The authors of this tutorial are co-authors on the ConvCNP paper.
```

It turns out that the type of generalisation we are looking for -- that the predictions of NPF members depend on the _relative_ position in input space of context and target points rather than the absolute one --
can be mathematically expressed as a property called _translation equivariance_.
Intuitively, translation equivariance (TE) states that if our observations are shifted in input space (which may be time, as in audio waveforms, or spatial coordinates, as in image data), then the resulting predictions should be shifted by the same amount.
This simple inductive bias, when appropriate, is extremely effective.
For example, convolutional neural networks (CNNs) were explicitly designed to satisfy this property {cite}`fukushima1982neocognitron`{cite}`lecun1989backpropagation`, making them the state-of-the-art architecture for spatially-structured data.

In {numref}`translation_equivariance_text`, we visualise translation equivariance in the setting of stochastic process prediction.
Here, we show stationary GP regression, which leads to translation equivariant predictions.
We can see that as the context set is _translated_, i.e., all the data points in $\mathcal{C}$ are shifted in input space by the same amount, so is the resulting predictive distribution.
To achieve spatial generalisation, we would like this property also to hold for neural processes.

```{figure} ../images/te_vis_complete.png
:width: 35em
:name: translation_equivariance_text
:alt: Translation equivariant mapping from dataset to predictive

Example of a translation equivariant mapping from a dataset to a predictive stochastic process.
```

```{admonition} Advanced$\qquad$Translation equivariance
---
class: hint, dropdown
---

More precisely, translation equivariance is a property of a _map_ between two spaces, where each space has a notion of translation defined on it (more precisely, there is a _group action_ of the translation group on each space). In the CNPF case, the input space $\mathcal{Z}$ is the space of finite datasets, such that each $Z \in \mathcal{Z}$ can be written as $Z = \{(x_n, y_n)\}_{n=1}^N$ for some $N \in \mathbb{N}$. The output space $\mathcal{H}$ is a space of continuous functions on $\mathcal{X}$, where $\mathcal{X}$ is the input domain of the regression problem (e.g., time, spatial position). Think of $\mathcal{H}$ as the set of all possible predictive mean and variance functions. The map acting from $\mathcal{Z}$ to $\mathcal{H}$ is then the CNPF member itself.

Let $\tau \in \mathcal{X}$ be a translation vector. We define _translations_ on each of these spaces as:

$$
\begin{align}
T \colon \mathcal{X} \times \mathcal{Z} \to \mathcal{Z}; & \qquad T_\tau Z = \{ (x_n + \tau, y_n) \}_{n=1}^N \\
T' \colon \mathcal{X} \times \mathcal{H} \to \mathcal{H}; & \qquad T'_\tau h(x) = h( x - \tau).
\end{align}
$$

Then, a mapping $f \colon \mathcal{Z} \to \mathcal{H}$ is said to be _translation equivariant_ if $f(T_\tau Z) = T'_\tau f(Z)$ for all $\tau \in \mathcal{X}$ and $Z \in \mathcal{Z}$. In other words, 'shifting then predicting' gives the same result as 'predicting then shifting'.
```

This provides the central motivation behind the ConvCNP {cite}`gordon2019convolutional`: _baking translation equivariance (TE) into the CNPF_, whilst preserving its other desirable properties.

To accomplish this, the ConvCNP maps datasets into a space of _continuous functions_ instead of a finite-dimensional vector space, as with the CNP and AttnCNP. These representations are known as _functional representations_ or _functional encodings_. To do this, the ConvCNP begins by performing a local functional encoding of each datapoint in the context set:

```{math}
:label: local_functional_encoding
\begin{align}
  R^{(c)}(x') = \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \psi \left( x^{(c)} - x' \right).
\end{align}
```
Here, $\psi$ is a function that maps the _distance between_ $x^{(c)}$ and $x'$ to a real number. $\psi$ is most often chosen to be an RBF: $\psi(r) = \exp(-\|r\|^2_2/ \ell^2)$, where $\ell$ is a learnable lengthscale parameter. You can think of this operation as simply placing Gaussian bumps down at every datapoint, similar to [Kernel Density Estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) or [Nadaraya–Watson kernel regression](https://en.wikipedia.org/wiki/Kernel_regression#Nadaraya%E2%80%93Watson_kernel_regression).

Note that the local encoding $R^{(c)}(x')$ is a _vector valued function_, since for each $x' \in \mathcal{X}$, it returns a vector in $\mathbb{R}^2$ --- it appends a constant 1 to the output $y^{(c)}$, which results in an additional _channel_. [^densityChannel] Intuitively, we can think of this additional channel -- referred to as the _density channel_ -- as keeping track of where data was observed in $\mathcal{C}$.

```{admonition} Note$\qquad$Density Channel
---
class: note, dropdown
---
To better understand the role of the density channel, consider a context point $(x^{(c)}, y^{(c)})$, with $y^{(c)} = 0$. Without the density channel, the local functional encoding for this point would be $R^{(c)}(x') = y^{(c)} \psi \left( x^{(c)} - x' \right)$, which is just the zero function. Hence, without a density channel, the encoder would be unable to distinguish between observing a context point with $y^{(c)} = 0$, and not observing a context point at all! With the density channel, the local functional encoding becomes $\begin{bmatrix} 1 \\ 0 \end{bmatrix} \psi \left( x^{(c)} - x' \right)$, which is no longer the zero function, and will hence contribute to the predictions. This turns out to be important in practice, as well as in the theoretical proofs regarding the expressivity of the ConvCNP.

```

We next describe the aggregator of the ConvCNP. In order to preserve translation equivariance, we want the aggregator to also be translation equivariant. A prime candidate for this in the family of deep learning architectures is the Convolutional Neural Network (CNN), which satisfies TE. There is however one issue: CNNs act on functions on a _discrete_ input space, but our local functional encodings are functions on a _continuous_ input space. To address this, we will _discretise_ the input to the CNN, and then smooth the output of the CNN to obtain a continuous function again. We first begin by adding up the local functional encodings, which is a _permutation invariant_ operation:

```{math}
:label: sum
\begin{align}
   \mathrm{sum}(x') = \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} R^{(c)}(x') = \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \psi(x^{(c)} - x').
\end{align}
```

```{admonition} Note$\qquad$Normalisation
---
class: note, dropdown
---
It is common practice to use the density channel to _normalise_ the output of the non-density channels (also called the _signal_ channels), so that the functional encoding becomes:

$$
\begin{align}
    \mathrm{density}(x') &= \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} \psi(x^{(c)} - x') \\
    \mathrm{signal}(x') &= \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} y^{(c)}\psi(x^{(c)} - x') \\
    \mathrm{sum}(x') &=  \begin{bmatrix} \mathrm{density}(x') \\ \mathrm{signal}(x') / \mathrm{density}(x') \end{bmatrix}
\end{align}
$$

Intuitively, what this does is ensure that the magnitude of the signal channel doesn't blow up if there are a large number of context points at the same spot. The density channel of the ConvCNP encoder can be seen as (a scaled version of) a kernel density estimate, and the normalised signal channel can be seen as Nadaraya-Watson kernel regression.
```

Next, we discretise the sum on an evenly spaced grid of input locations $\{ x^{(u)} \}_{u=1}^U$:

```{math}
:label: discretisation
\begin{align}
   \mathrm{sum}(x') \to \{ \mathrm{sum}(x^{(u)}) \}_{u=1}^U
\end{align}
```


```{admonition} Important
---
class: attention
---
This discretisation means that the resulting ConvCNP can only be approximately TE, where the quality of the approximation is controlled by the number of points $U$.
If the spacing between the grid points is $\Delta$, the ConvCNP would not be expected to be equivariant to shifts of the input that are smaller than $\Delta$.
```

Next, we can pass the discretised sum through a CNN, and smooth the output to obtain our functional encoding for the entire context set, $R(x')$:

```{math}
:label: CNN
\begin{align}
\{f(x^{(u)})\}_{u=1}^U &= \mathrm{CNN}(\{ \mathrm{sum}(x^{(u)}) \}_{u=1}^U) \\
R(x') &= \sum_{u=1}^U f(x^{(u)}) \psi(x^{(u)} - x'),
\end{align}
```

where $\psi$ is another Gaussian RBF. Note that $R(x')$ is a function on a _continuous_ input domain: it can be evaluated at any input location (e.g. any time or spatial position). Finally, to obtain the predictive mean and variance at a target input $x^{(t)}$, we evaluate the functional encoding at $x^{(t)}$, and pass the result into the decoder, which is an MLP:

```{math}
:label: final_decoder_layer
\mu^{(t)}, \log \sigma^{2(t)} = \mathrm{MLP}(R(x^{(t)}))
```

```{admonition} Note$\qquad$MLP Decoder
---
class: note, dropdown
---
In the original ConvCNP paper, the MLP decoder is sometimes omitted, with $R(x')$ directly parameterising the predictive mean and variance.
```

Putting everything together, we can define the ConvCNP using the following design choices (illustrated in {numref}`computational_graph_ConvCNPs_text`):
* The local encoding maps context set points to functions via {numref}`local_functional_encoding`.
* The aggregation function is a sum, followed by a discretisation, CNN, and smoothing ({numref}`sum`, {numref}`discretisation`, {numref}`CNN`).
* Finally, the decoder is a pointwise MLP ({numref}`final_decoder_layer`).

Note that the separation of the ConvCNP into local encoding, aggregator and decoder is somewhat arbitrary. You could also view summation as the aggregator, and the CNN as part of the decoder, which is the view presented in the original ConvCNP paper.

```{admonition} Note$\qquad$On-the-grid ConvCNP
---
class: note, dropdown
---
The architecture described so far is referred to as the 'off-the-grid' ConvCNP in the original paper, since it can handle input observations at any point on the $x$-axis. However, we sometimes have situations where inputs end up regularly spaced on the $x$-axis. A prime example of this is in image data, where the input image lives on a regularly-spaced 2D grid. For this case, the authors propose the 'on-the-grid' ConvCNP, which is a simplified version that is easier to implement.

In the on-the-grid ConvCNP, the input data already lives on a discretised grid. Hence, after appending a density channel, this can immediately be fed into a CNN, without the need for RBF smoothing or discretisation. Suppose we have an image, represented as an $H \times W$ matrix. Within this image, we have $C$ observed pixels and $HW - C$ unobserved pixels. We represent an observed pixel with the vector $[1, y^{(c)}]^T$, and an unobserved pixel with the vector $[0, 0]^T$. As before, the first element of this vector is the density channel, and indicates that a datapoint has been observed.

To implement this, let the input image be $I \in \mathbb{R}^{H \times W}$. Let $M \in \mathbb{R}^{H \times W}$ be a mask matrix, with $M_{i,j} = 1$ if the pixel at location $(i,j)$ is in the context set, and $0$ otherwise. Then we can compute the density channel as $\mathrm{density} = M$ and the signal channel as $\mathrm{signal} = I \odot M$, where $\odot$ denotes element-wise multiplication. We then stack these matrices as $[\mathrm{density}, \mathrm{signal}]^T$. This can then be passed into a CNN, and the CNN can output one channel for the predictive mean, and another for the log predictive variance.

```

<!-- To accomplish this, the ConvCNP designs the encoder and decoder to each be translation equivariant. Recall that the encoder maps the context set to a representation $R$. For the CNP and AttnCNP, $R$ was a finite-dimensional vector. However, it is unclear what it means to 'translate' this abstract representation vector. Hence, the ConvCNP instead represents each context set $\mathcal{C}$ with a _continuous function_ $R(\cdot) : \mathcal{X} \to \mathbb{R}^{d_r}$, where $\mathcal{X}$ is the input domain (i.e., the $x$-axis, or pixel location). We will refer to representations of this form as _functional representations_ or _functional encodings_.

The encoder will be designed with the property that if the context set is shifted in input space, the functional encoding $R(\cdot)$ will be shifted by the same amount. To obtain the predictive mean and variance functions $\mu(\cdot), \sigma^2(\cdot)$, the functional encoding $R(\cdot)$ is passed through a CNN. Since CNNs are also translation equivariant, the entire ConvCNP is translation equivariant.

The main question then becomes: how should we map contexts sets to functional encodings?
One way to view this is through the previously discussed 2-step process of local encoding followed by aggregation.
However, now the local encoding for each data point in $\mathcal{C}$ is a _function_.
We can express these _local_ functional encodings as

```{math}
:label: local_functional_encoding
\begin{align}
  R^{(c)}(x') = \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \psi \left( x^{(c)} - x' \right).
\end{align}
```

Here, $\psi$ is a function that maps the _distance between_ $x^{(c)}$ and $x'$ to a real number. $\psi$ is most often chosen to be an RBF kernel: $\psi(r) = \exp(-r^2/ \ell^2)$, where $\ell$ is a learnable lengthscale parameter. Note that the local encoding $R^{(c)}(\cdot)$ is a _vector valued function_, since for each $x' \in \mathcal{X}$, it returns a vector in $\mathbb{R}^2$ --- it appends a constant 1 to the output $y^{(c)}$, which results in an additional _channel_. [^densityChannel] Intuitively, we can think of this additional channel -- referred to as the _density channel_ -- as keeping track of where data was observed in $\mathcal{C}$.

Now, the aggregation function consists of two parts: begins by simply summing of all of the local embeddings, which results in yet another function!
So, the functional encoding for the entire context set $\mathcal{C}$ can be written as

```{math}
:label: functional_encoding
\begin{align}
   R(x') = \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} R^{(c)}(x') = \sum_{(x^{(c)}, y^{(c)}) \in \mathcal{C}} \begin{bmatrix} 1 \\ y^{(c)} \end{bmatrix} \psi(x^{(c)} - x').
\end{align}
```


```{admonition} Important
---
class: attention
---
Although the output of {numref}`functional_encoding` is an actual function, only the values of the functional representation at a finite number of equally-spaced grid locations $\{x^{(u)}\}_{u=1}^{U}$ can be stored on a computer for dowstream processing.
We say the resulting function is *discretised* as it is represented by a finite set $\{(x^{(u)}, R(x^{(u)}))\}_{u=1}^{U}$.

This discretisation means that the resulting ConvCNP can only be approximately TE, where the quality of the approximation is controlled by the number of points $U$.
If the spacing between the grid points is $\Delta$, the ConvCNP would not be expected to be equivariant to shifts of the input that are smaller than $\Delta$.
The upside is that the resulting (discrete) function can be processed by standard CNNs, as these operate on _discrete_ inputs.
```

After computing the discretised functional encoding, it is then passed through a CNN (1D convolutions for 1D regression, 2D convolutions for image data).
Finally, the discrete output of the CNN ($f_\mu(x^{(u)})$ and $f_{\sigma^2}(x^{(u)})$ for the mean and variance functions, respectively) is again smoothed using a Gaussian bump function ($\psi_\rho$).
Hence the predictive mean and variance can be evaluated at _any_ target location $x^{(t)}$:

```{math}
:label: final_decoder_layer
\begin{align}
   \mu(x^{(t)}) = \sum_{u=1}^U f_\mu(x^{(u)}) \psi_\rho(x^{(t)} - x^{(u)});
   \quad
   \log \sigma^2(x^{(t)}) = \sum_{u=1}^U f_\sigma(x^{(u)}) \psi_\rho(x^{(t)} - x^{(u)}),
\end{align}
```

Putting everything together, we can define the ConvCNP using the following design choices (illustrated in {numref}`computational_graph_ConvCNPs_text`):
* The local encoding maps context set points to functions via {numref}`local_functional_encoding`.
* The aggregation function is simply the sum of the local functional encodings ({numref}`functional_encoding`), followed by a discretisation to evaluate the resulting function at a fixed grid of locations.
* Finally, the decoder is a CNN, followed by an additional layer to evaluate the means and standard deviations at the target locations ({numref}`final_decoder_layer`).

```{admonition} Note$\qquad$Point-wise MLP decoders
---
class: note, dropdown
---
In practice, it is also possible to add a point-wise MLP to the decoder, after evaluating the CNN output at a target location $x^{(t)}$.
In this case, the MLP maps the "smoothed" CNN channels $f(x^{(t)})$ to the mean and variance functions.
Gordon et al. did not employ such an MLP, but we do so in our experiments in this post.
``` -->

<!---
Putting everything together, the ConvCNP computes a target-dependent representation $R^{(t)}$ as follows (illustrated in {numref}`computational_graph_ConvCNPs_text`):
1. Map the context set to a functional representation $R(\cdot)$ using SetConv.
2. Discretise $R(\cdot)$.
3. Process the discretised $R(\cdot)$ with a CNN.
4. Get a target dependent representation $R^{(t)}$ by evaluating the functional representation at each target location $R(x^{(t)})$.
-->

```{figure} ../images/computational_graph_ConvCNPs.svg
---
height: 300em
name: computational_graph_ConvCNPs_text
alt: Computational graph ConvCNP
---
Computational graph of ConvCNP.
```


TO DO: ADD GIF OF CONVCNP

```{admonition} Advanced$\qquad$ConvDeepSets
---
class: dropdown, hint
---
Similar to how CNPs make heavy use of "DeepSet" networks, ConvCNPs rely on the _ConvDeepSet_ architecture which was also proposed by Gordon et al. {cite}`gordon2019convolutional`.
ConvDeepSets map sets of the form $Z = \{(x_n, y_n)\}_{n=1}^N$, $N \in \mathbb{N}$ to a space of functions $\mathcal{H}$, and have the following form

$$
\begin{align}
f(Z) = \rho(E (Z)); \qquad E(Z) = \sum_{(x,y) \in Z} \phi(y) \psi(\cdot - x) \in \mathcal{H},
\end{align}
$$

for appropriate functions $\phi$, and $\psi$, and $\rho$ translation equivariant.
The key result of Gordon et al. is to demonstrate that not only are functions of this form translation equivariant, but under mild conditions, _any_ continuous and translation equivariant function on sets $Z$ has a representation of this form.
This result is analagous to that of Zaheer et al., but extended to translation equivariant functions on sets.
<!-- Gordon et al. rely on the universal approximators for $phi$, the notion of _interpolating kernels_ for $\psi$, and specify $\rho$ as universal approximators of continuous _and_ translation equivariant functions between function spaces. -->

ConvCNPs then define the parameterisations of the mean and variance functions using ConvDeepSets.
In particular, ConvCNPs specify $\phi \colon y \mapsto (1, y)^T$, $\rho$ as a CNN, and $\psi$ as an RBF kernel, and use ConvDeepSets to parameterise the predictive mean and standard deviation functions.
Thus, ConvDeepSets motivate the architectural choices of the ConvCNP analagously to how DeepSets can be used to provide justification for the CNP architecture.
```


<!---
```{admonition} Advanced$\qquad$Representation Theorem
---
class: dropdown, hint
---
{cite}`gordon2019convolutional` demonstrate that _any_ (continuous) permutation invariance and TE function can be represented by a ConvCNP.
The proof relies on first extending the DeepSets work of {cite}`zaheer2017deep` to include TE as well.
In the {doc}`Theory<Theory>` chapter, we provide a sketch of this proof.
```
-->

```{admonition} Note$\qquad$Computational Complexity
---
class: dropdown, note
---
Computing the discrete functional representation requires considering $C$ points in the context set for each discretised function location which scales as $\mathcal{O}(U*C)$.
Similarly, computing the predictive at the target inputs scales as $\mathcal{O}(U*T)$. Finally, if the convolutional kernel has width $K$, each convolution scales as $\mathcal{O}(U*K)$ (here we are ignoring the added complexity due to multiple channels). Hence the computational complexity of inference in a ConvCNP is thus $\mathcal{O}(U(C+K+T))$. In the off-the-grid ConvCNP, the computational complexity is simply $\mathcal{O}(U*K)$, where $U$ is the number of pixels for image data.

This shows that there is a trade-off: if the number of discretisation points $U$ is too large then the computational cost will not be manageable, but if it is too small then ConvCNP will be only very 'coarsely' TE.
```

Now that we have constructed a translation equivariant member of the CNPF, we can test it in the more challenging extrapolation regime.
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


````{admonition} Caution$\qquad$Receptive Field
---
class: dropdown, warning
---
The periodic kernel example is a little misleading, as the ConvCNP does not recover the underlying GP predictions everywhere.
In fact, we know that it cannot exactly recover the underlying process.
Indeed, it can only model _local_ periodicity because it has a bounded _receptive field_ --- the
size of the input region that can affect a particular output. See [this Distill article](https://distill.pub/2019/computing-receptive-fields/) for a further explanation of CNN receptive fields.
This is best seen when considering a much larger target interval ($[-2,14]$ instead of $[0,4]$):

```{figure} ../gifs/ConvCNP_periodic_large_extrap.gif
---
width: 35em
name: ConvCNP_periodic_large_extrap_text
alt: ConvCNP on single images
---
Large extrapolation (red dashes) of posterior predictive of ConvCNPs (Blue) and the oracle GP (Green) with periodic kernel.
```

In fact, this is true for any of the GPs above, all of which have "infinite receptive fields" --- in principle, an observation at one point affects the predictions along the entire $x$-axis. This means that no model with a bounded receptive field can _exactly_ recover the GP predictive distribution everywhere.
In practice however, most GPs (e.g., with RBF and Matern kernels) have a finite length-scale, and points much further apart than the length-scale are, for all practical purposes, independent.

This discussion alludes to one of the key design choices of the ConvCNP, which is the size of its receptive field.
Note that, unlike standard CNNs, the resulting receptive field does not only depend on the CNN architecture, but also on the granularity of the discretisation employed on the functional representation.
````


Let's now examine the performance of the ConvCNP on more challenging image experiments.
As with the AttnCNP, we consider CelebA and MNIST reconstruction experiments, but also include the Zero-Shot Multi-MNIST (ZSMM) experiments that evaluate the model's ability to generalise beyond the training data.

```{figure} ../gifs/ConvCNP_img.gif
---
width: 45em
name: ConvCNP_img_text
alt: ConvCNP on CelebA, MNIST, ZSMM
---

Posterior predictive of an ConvCNP for CelebA, MNIST, and ZSMM.
```

From {numref}`ConvCNP_img_text` we see that the ConvCNP performs quite well on all datasets when the context set is large enough and uniformly sampled, even when extrapolation is needed (ZSMM).
However, performance is less impressive when the context set is very small or when it is structured, e.g., half images.
In our experiments we find that this is more of an issue for the ConvCNP than the AttnCNP ({numref}`AttnCNP_img`); we hypothesize that this happens because the effective receptive field of the former is too small.

```{admonition} Note$\qquad$Effective Receptive Field
---
class: dropdown, note
---
We call the *effective* receptive field, the empirical receptive field of a *trained* model rather than the theoretical receptive field for a given architecture.
For example if a ConvCNP always observes many context points during training, then every target point will be close to some to context points and the ConvCNP will thus not need to learn to depend on context points that are far from target points.
The size of the effective receptive field can thus be increased by reducing the size of the context set seen during training, but this solution is somewhat dissatisfying in that it requires tinkering with the training procedure.

In contrast, this is not really an issue with AttnCNP which both always has to attend to all the context points, i.e., it has an ``infinite'' receptive field.
```

Although the previous plots look good, you might wonder how such a model compares to standard interpolation baselines.
To answer this question we will look at larger images to see the more fine grained details.
Specifically, let's consider a ConvCNP trained on $128 \times 128$ CelebA:

```{figure} ../gifs/ConvCNP_img_baselines.gif
---
width: 25em
name: ConvCNP_img_baselines
alt: ConvCNP and baselines on CelebA 128
---

ConvCNP and Nearest neighbour, bilinear, bicubic interpolation on CelebA 128.
```

{numref}`ConvCNP_img_baselines` shows that the ConvCNP performs much better than baseline interpolation methods. Having seen such encouraging results, as well as the decent zero shot generalisation capability of the ConvCNP on ZSMM, it is natural to want to evaluate the model on actual images with multiple faces with different scales and orientations:

```{figure} ../images/ConvCNP_img_zeroshot.png
---
width: 20em
name: ConvCNP_img_zeroshot_text
alt: Zero shot generalization of ConvCNP to a real picture
---
Zero shot generalization of a ConvCNP trained on CelebA and evaluated on Ellen's selfie. We also show a baseline bilinear interpolator.
```

From {numref}`ConvCNP_img_zeroshot_text` we see that the model trained on single faces is able to generalise reasonably well to real world data in a zero shot fashion. One possible application of the ConvCNP is increasing the resolution of an image.
This can be achieved by querying positions "in between" pixels.

```{figure} ../images/ConvCNP_superes_baseline.png
---
width: 30em
name: ConvCNP_superes_baseline_text
alt: Increasing image resolution with ConvCNP
---
Increasing the resolution of $16 \times 16$ CelebA to $128 \times 128$ with a ConvCNP and a baseline bilinear interpolator.
```

{numref}`ConvCNP_superes_baseline_text` demonstrates such an application.
We see that the ConvCNP can indeed be used to increase the resolution of an image better than the baseline bilinear interpolator, even though it was not explicitly trained to do so!


```{admonition} Note
---
class: note
---
Model details, training and many more plots are available in the {doc}`ConvCNP Notebook <../reproducibility/ConvCNP>`
```

(issues-cnpfs)=
## Issues With the CNPF

Let's take a step back.
So far, we have seen that we can use the factorisation assumption to construct members of the CNPF, perhaps the simplest of these being the CNP.
Our first observation was that while the CNP can predict simple stochatic processes, it tends to underfit when the processes are more complicated.
We saw that this tendency can be addressed by adding appropriate inductive biases to the model.
Specifically, the AttnCNP significantly improves upon the CNP by adding an attention mechanism to generate target-specific representations of the context set.
However, both the CNP and AttnCNP fail to make meaningful predictions when data is observed outside the training range.
Finally, we saw how including translation equivariance as an inductive bias led to accurate predictions that generalised elegantly to observations outside the training range.

Let's now consider more closely the implications of the factorisation assumption, along with the Gaussian form of predictive distributions.
One immediate consequence of using a Gaussian likelihood is that we cannot model multi-modal predictive distributions.
To see why this might be an issue, consider making predictions for the MNIST reconstruction experiments.

```{figure} ../images/ConvCNP_marginal.png
---
width: 20em
name: ConvCNP_marginal_text
alt: Samples from ConvCNP on MNIST and posterior of different pixels
---

Predictive distribution of a ConvCNP on an entire MNIST image (left) and marginal predictive distributions of some pixels (right).
```

Looking at {numref}`ConvCNP_marginal_text`, we might expect that sampling from the predictive distribution for an unobserved pixel would sometimes yield completely white values, and sometimes completely black --- depending on whether the sample represents, for example, a 3 or a 5.
However, a Gaussian distribution, which is unimodal (see {numref}`ConvCNP_marginal_text` right), cannot model this multimodality.

<!---
One possible solution to this problem might be to employ some other parametric distribution that enables multimodality, for example, a mixture of Gaussians.
While this may solve some issues, we can generalise this point to say that the CNPF requires specifying _some parametric form of distribution_.
Ideally, what we would like is some parametrisation of the NPF that enables us to recover _any_ form of marginal distribution.
-->

The other major restriction is the factorisation assumption itself.
First, the model cannot model any dependencies in the predictive distribution over multiple target points.
For example imagine that we are modelling samples from a GP.
If the model is making predictions at two target locations that are "close" on the $x$-axis, it seems reasonable that whenever it predicts the first output to be "high", it would predict something similar for the second, and vice versa.
Yet the factorisation assumption prevents this type of correlation from occurring.
Another way to view this is that the CNPF cannot produce _coherent_ samples from its predictive distribution.
In fact, sampling from the posterior corresponds to adding independent noise to the mean at each target location, resulting in samples that are discontinuous and look nothing like the underlying process:

```{figure} ../images/ConvCNP_rbf_samples.png
---
width: 35em
name: ConvCNP_rbf_samples_text
alt: Sampling from ConvCNP on GP with RBF kernel
---
Samples form the posterior predictive of a ConvCNP (Blue), and the predictive distribution of the oracle GP (Green) with RBF kernel.
```

Similarly, sampled images from a member of the CPNF are not coherent and look like random noise added to a picture:

```{figure} ../images/ConvCNP_img_sampling.png
---
width: 45em
name: ConvCNP_img_sampling_text
alt: Sampling from ConvCNP on CelebA, MNIST, ZSMM
---

Samples from the posterior predictive of an ConvCNP for CelebA, MNIST, and ZSMM.
```


```{admonition} Note$\qquad$Thompson Sampling
---
class: note, dropdown
---
This inability to sample from the predictive may inhibit the deployment of CNPF members from several application areas for which it might otherwise be potentially well-suited.
One such example is the use of Thompson sampling algorithms for e.g., contextual bandits or Bayesian optimisation, which require a model to produce samples.
```

In the next chapter, we will see one approach to solving both these issues by treating the representation as a latent variable. This leads us to the _latent_ Neural Process family (LNPF).

[^densityChannel]: Alernatively, these would be vectors in $\mathbb{R}^4$ if we were modelling RGB images.

[^AttnCNP]: {cite}`kim2019attentive` only introduced the latent variable model, but one can easily drop the latent variable if not needed.
