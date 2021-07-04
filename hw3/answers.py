r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['seq_len'] = 80
    hypers['h_dim'] = 128
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.2
    hypers['learn_rate'] = 1e-3
    hypers['lr_sched_factor'] = 1e-1
    hypers['lr_sched_patience'] = 4
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "to be or not to be"
    temperature = 5e-6
    # raise NotImplementedError()
    # ========================
    return start_seq, temperature

part1_q1 = r"""
**Your answer:**

Using the whole text may lead to over fitting, where the model will memorize the text and fail to generalize. 
Splitting the corpus into sequences prevents this problem since it allows the model to see different 
character sequences with various orders and with various hidden state memory in different epochs.

"""

part1_q2 = r"""
**Your answer:**

The hidden state does not depend on the length of the sequence, thus when generating text it may be of a different 
length than the sequence's length.

"""

part1_q3 = r"""
**Your answer:**
When training the network we pass the hidden state across batches, hoping that it will benefit the learning 
of the the model by storing relevant  data in the hidden state. 
to preserve the order of the original text, we need to pass the most relevant hidden state across training batches
"""

part1_q4 = r"""
**Your answer:**

1. the temperature parameter controls the diffusion of the distribution. as the temperature get higher the distribution 
become more uniform while lowering the temperature, we make the distribution more spiky.
since we want the to choose chars that are more likely by the model, we would like to lower the temperature 
to give those chars a higher probability to be chosen.


2. when the temperature is very high the distribution become uniform, when it can be see from the hot softmax formula:  
$\text{hot_softmax}_T(y) = \frac{e^{y/T}}{\sum_k e^{y_k/T}}$. 
when T is very high then $e^{y/T}\rightarrow 1$ and the distribution  become uniform.


3. when the temperature is very low, the factor of ${1/T}$ become very large and emphasize more the chars with higher 
probabilities. wich we can see in the hot softmax formula: $\text{hot_softmax}_T(y) = \frac{e^{y/T}}{\sum_k e^{y_k/T}}$. 
which make the probability distribution more spiky.

"""
# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=4, h_dim=128, z_dim=256, x_sigma2=0.1, learn_rate=0.0005, betas=(0.9, 0.999),)

    # raise NotImplementedError()
    # ========================
    return hypers


part2_q1 = r"""

$\sigma^2$ determines the variance of the normal distribution of the likelihood 
$p _{\bb{\beta}}(\bb{X} | \bb{Z}=\bb{z}) = \mathcal{N}( \Psi _{\bb{\beta}}(\bb{z}) , \sigma^2 \bb{I} )$.
Increasing its value will lead to a wider normal distribution 
which yields more variability in sampling from latent space while generating samples using the decoder,
meaning we will get samples that may 'look different' from the samples in the original dataset.
On the other hand, decreasing its value will yield a narrower distribution, causing the samples to be more similar to 
the ones from the dataset. 

Another view of  $\sigma^2$ is it controls to what extent does the data-reconstruction loss effects the total loss, 
in comparasent with the KL-divergence loss. The larger $\sigma^2$ is, the less effect the data-reconstruction loss has. 
Meaning the variance of the generated data will be large. And the smaller $\sigma^2$ is, the more biased the generated 
data will be towards existing data.
 

"""

part2_q2 = r"""

1. The reconstruction loss, given by
$$
L_{reconstruciton} =
\frac{1}{\sigma^2 d_x} \left\| \bb{x}- \Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  +
\bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right) \right\| _2^2
$$

tries to make the generated data as close as possible (under L_2 norm) to the original input data. As when minimizing it, 
we minimize the difference between the real observation $x$ and the VAE output (encoded and decoded) 
$\Psi _{\bb{\beta}}\left(  \bb{\mu} _{\bb{\alpha}}(\bb{x})  +
\bb{\Sigma}^{\frac{1}{2}} _{\bb{\alpha}}(\bb{x}) \bb{u}   \right)$


The KL-loss, given by
$L_{KL}=\mathrm{tr}\,\bb{\Sigma} _{\bb{\alpha}}(\bb{x}) +  \|\bb{\mu} _{\bb{\alpha}}(\bb{x})\|^2 _2 - d_z - \log\det \bb{\Sigma} _{\bb{\alpha}}(\bb{x}),
$

tried to make the encoder correctly approximate the posterior distribution $p(z|X)$ and 
hence generate images that look like the real data.


2. The idea in VAE is to estimate the evidence distribution $p(X)$. As
 this can be very hard to compute, we will try to maximize a lower bound on $log(p(X))$ as explained in the hw.
The lower bown is given by  $ \log p(\bb{X}) \ge \mathbb{E} _{\bb{z} \sim q _{\bb{\alpha}} }\left[ \log  p _{\bb{\beta}}(\bb{X} | \bb{z}) \right]
-  \mathcal{D} _{\mathrm{KL}}\left(q _{\bb{\alpha}}(\bb{Z} | \bb{X})\,\left\|\, p(\bb{Z} )\right.\right)
$

where $
\mathcal{D} _{\mathrm{KL}}(q\left\|\right.p) =
\mathbb{E}_{\bb{z}\sim q}\left[ \log \frac{q(\bb{Z})}{p(\bb{Z})} \right]
$
is the KL-divergence.

So when minimizing the KL-loss, we're finding a Gaussian distribution $q(Z)$ that is as close as possible to $p(Z)$.
Thus making the lower bound tighter. In addition, the KL-loss term forces $q$ to be a distribution rather than a 
point-mass (that would minimize the reconstruction loss). 

3. The benifit of this effect is allows controlling the variance of the latent space distribution. This can be seen 
as part of the tradeoff between the reconstruction-loss and the KL-loss as we explained in q_1.

"""

part2_q3 = r"""
In general, we don't know what is the evidence distribution $p(X)$  (very complex and infeasible to compute).
Our goal is to learn an estimation of it in order be able to generate new data-points.
We do so by maximizing a lower bound (as explained in q_2) and aim to get it as tight as possible so that we'd get 
an estimation of $p(X)$ that's as accurate as possible.

"""

part2_q4 = r"""

We model the log of the latent-space variance corresponding to an input,  $\sigma^2_{\alpha}$ ,
instead of directly modelling the variance to make the training process more numerically stable.

While $\sigma^2_{\alpha}$ 's values are by definition positive real numbers $[0, \inf]$ (usually close to zero). 
Using the log transformation(and then using the exponent of the log) not only enforces $\sigma^2_{\alpha}$'s 
values to be positive ($log(\sigma^2_{\alpha})$ 's values can be any number in $[-\inf, \inf]$) but also 'expands' the 
region close to 0, which gives the model numerical stability (as the representation of very small positive values as 
floating points might not be accurate enough). 

In addition, the log function is differentiable in all the range it's defined upon, which allows using the derivative easily 
(Giving another aspect in which using this transformation increases the numerical stability.)

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    hypers = dict(
        batch_size=8,
        z_dim=256,
        data_label=1,
        label_noise=0.3,
        discriminator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,

            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="Adam",  # Any name in nn.optim like SGD, Adam
            lr=0.0005,
            # You an add extra args for the optimizer here
        ),
    )

    # ========================
    return hypers


part3_q1 = r"""

The GAN model consists of 2 parts: generator and discriminator. The goal of the generator is to generate 'realistic' 
data that would fool the discriminator into classifying it as real, and the discriminator's goal is to decide whether a 
given datapoint is real or not (generated).

Training the GAN model is done by optimizing the losses of each part, one after another for each batch.

The discriminator is trained by sampling a datapoint using the generator and checking what is the discriminator's 
classification and updating the discriminator based on the discriminator-loss. In this part we do not maintain the 
gradients (with_grad=False) as we don't need to update the generator's weights during backpropagation.  

The generator is trained by sampling a datapoint (generating it) and then showing it to the discriminator and updating 
the generator based on the generator-loss. In this part we do maintain the gradients (with_grad=True) as we do want the 
generator's weights to be updated during backpropagation.  

"""

part3_q2 = r"""
1. No. In GANs, the evaluation of the generator loss is done using the discriminator.
 Assume the discriminator performs poorly, meaning it is easily fooled. In such scenario,  
 the generator loss can increase even though it is not generating good samples. (Note: the other direction holds as well
 meaning that the evaluation of the discriminator depends on the generator's performance)


2. If the discriminator loss remains at constant value while the generator loss decreases, 
it means that the model is failing to converge. 
As this means the generator is able to fool the discriminator more ofter as we train 
but the discriminator is not getting better at identifying non-real data. And as both parts' evaluation depends 
on one another, this means the discriminator is not improving and not good enough to perform well.

"""

part3_q3 = r"""
Comparing the VAE results with the GAN results we can see that the VAE results are more blurry with less sharp edges, 
while the GAN results are sharper and more detailed.
We think this is the expected result as the VAE loss has a term of reconstruction loss,
forcing the output to be similar to the input in terms of MSE loss which results in blurry images and smooth edges
and making the generated images more similar to each other. On the other hand in GAN, the generator does not have
'direct access' to real images but learns how those should look through the decisions of the discriminator, 
forcing it's predictions to be more realistic (because otherwise it would be easy for the discriminator to 
detect generated images)

"""

# ==============
