---
title: "Generalization of Diffusion Models Arises with a Balanced Representation Space"
permalink: /
layout: single
classes: wide
---
<p class="button-row">
<a class="btn btn--info" href="https://arxiv.org/abs/2512.20963">arXiv</a>
<a class="btn btn--success" href="{{ site.github.repository_url }}">Code</a>
<a class="btn btn--warning" href="https://drive.google.com/file/d/12A0cRa1vq_kCqEHYl_2rMLuMIZv64RmV/view?usp=drive_link">Slides</a>
</p>

---
<p class="lead-italic"><em>What are we talking about when we talk about generalization?</em></p>
Ultimately, generalization is the **implicit alignment** between **neural networks** and the **underlying distribution** $$p_{\mathrm{gt}}$$ (i.e., human-defined data and perception).

<img src="{{ '/assets/figures/network_learns.png' | relative_url }}" width="60%" style="display:block;margin:auto;" />
<p style="text-align:center;"><strong>Generalization as alignment.</strong> The network learns beyond the finite training set to approximate ground truth.</p>

In diffusion models, this means generating meaningful images never present in the training set by learning a denoiser from empirical samples $$\bm{x}_{i=1\dots n}$$. A standard training objective is:

$$
\frac{1}{T}\sum_{t=0}^{T}
\mathbb{E}_{\bm{x}\sim p_{\mathrm{gt}},\,\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})}
\!\left[\big\|\bm{f}_{\bm{\theta}}(\bm{x}+\sigma_t \bm{\epsilon},t)-\bm{x}\big\|^2\right].
$$

If we learn an approximate $$\bm{f}_{\mathrm{emp}}$$, sampling will starts from noise and iteratively denoises to meaningful images.

This striking ability is *not* simply because neural networks can approximate arbitrary functions. Otherwise, training would routinely collapse to an empirical solution that memorizes training samples:


---
<p class="lead-italic"><em>Understanding this requires looking into networks.</em></p>
We study parameterized denoisers trained with gradient descent in a minimal setup: a two-layer ReLU network under a single noise level. Since it is also a denoising autoencoder, we call it **ReLU-DAE**.

$$
\bm{f}_{\bm{W}_2,\bm{W}_1}(\bm{x})
= \bm{W}_2\bm{h}(\bm{x})
= \bm{W}_2\,[\bm{W}_1^\top \bm{x}]_+.
$$

We prove a clean correspondence:  
(i) *memorization* $$\Leftrightarrow$$ storing raw samples in the weights, approximating $$\bm{f}_{\mathrm{emp}}$$;  
(ii) *generalization* $$\Leftrightarrow$$ learning local data statistics, approximating $$\bm{f}_{\mathrm{gt}}$$; and  
(iii) a *hybrid regime* due to data imbalance.

<img src="{{ '/assets/figures/teaser.png' | relative_url }}" width="70%" style="display:block;margin:auto;" />
<p style="text-align:center;"><strong>Three regimes in ReLU-DAE learning.</strong> Memorization (left), hybrid (center), and generalization (right)</p>

---
<p class="lead-italic"><em>Representation learning follows naturally.</em></p>
Memorized samples align perfectly with stored structures and produce *spiky* representations: think of a strong single-neuron response or retrieval of a specific training example.  
Generalized samples align with a broader set of structures, yielding *balanced* representations that compose across neurons and reflect the underlying distribution, serving as a coordinate for the image manifold.

<div style="display:flex; gap:20px; align-items:flex-start; flex-wrap:wrap;">
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/celeba_rep.png' | relative_url }}" style="width:100%;" />
  </div>
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/imagenet_rep.png' | relative_url }}" style="width:100%;" />
  </div>
  <div style="flex:1; min-width:240px;">
    <img src="{{ '/assets/figures/LAION_rep.png' | relative_url }}" style="width:100%;" />
  </div>
</div>

<p style="text-align:center;"><strong>Same signature in real diffusion models.</strong> The spiky-vs-balanced separation persists in large models.</p>

---

Generalized representations can also be manipulated to change the final generation.
<div style="display:flex; gap:16px; flex-wrap:wrap; justify-content:center;">
  <div style="flex:1; min-width:260px; max-width:360px;">
    <img src="{{ '/assets/figures/man_age.png' | relative_url }}" style="width:100%;" />
    <p style="text-align:center;"><strong>+Old (Gen.)</strong></p>
  </div>
  <div style="flex:1; min-width:260px; max-width:360px;">
    <img src="{{ '/assets/figures/mem_dt_age.png' | relative_url }}" style="width:100%;" />
    <p style="text-align:center;"><strong>+Old (Mem.)</strong></p>
  </div>
</div>

<p style="text-align:center;"><strong>Image editing via representation steering.</strong> Works for generalized samples but not for memorized samples.</p>

---
<p class="lead-italic"><em>Our theory starts from a simple two-layer network.</em></p>
We believe it reflects a fundamental mechanism in deep models: they project noisy inputs onto learned low-dimensional structure, *arranging visually similar inputs into similar activations* (via ReLU gating in our theory). 

This arrangement underlies their **compressing and denoising nature**, and aligns strongly with human perception. Internally, it is reflected as representation learning; empirically, learning a balanced and semantic representation space is a strong indicator of generalization.
