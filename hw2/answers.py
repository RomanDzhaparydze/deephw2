r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1) Y has a shape of 64×512, X has a shape of 64×1024
64×1024. The Jacobian captures how each output element depends on each input element, so its shape is: 64x512x64x1024. The jacobian is sparse, becauseЖ
∂Y(i,j)/∂X(i,j)=0 for i!=j, thus Jacobian is kind of block-diagonal matrix.
We do not need to materialize the jacobian itself, but we can use chain rule δX=δYxW^T.
2) W has a shape of 512x1024, Y has shape of 64x512, thus the jacobian ∂Y/∂W has shape of 64x512x512x1024. The jacobian is not sparse, every element of Y depends on multiple elements of W, due to the nature of fully-connectivity. We still do not need to materialize the jacobian it self and can use chain rule: δW=δY^TxX 
"""

part1_q2 = r"""
**Your answer:**

Yes, backpropagation is necessary to efficiently train neural networks using gradient-based optimization. It allows the computation of gradients of the loss function w.r.t. all parameters in a very computationally efficient manner. Gradient-based optimization, SGD-based algorithms use such gradients for updating the weights and biases of the model. Backpropagation is a way of propagating the error backwards through the network, layer by layer, using the chain rule. Its time complexity is linear in the number of layers, while without backpropagation, it would be very slow and computationally expensive to compute the gradients directly.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.125
    lr = 0.045
    reg = 0.029
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.1
    lr_momentum = 0.0075
    lr_rmsprop = 0.00015
    reg = 0.0001

    # wstd = 0.1
    # lr_vanilla = 0.03
    # lr_momentum = 0.1
    # lr_rmsprop = 0.1
    # reg = 0.001
    
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0025
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. The graphs of no-dropout vs dropout match with what we've expected to see:
The no-dropout model has high capacity and learns to memorize the whole training data, which lead to overfitting
The dropout models however did not overfit, because dropout acts as a regularizer, which reduces overfitting and relying to much on some specific neurons.
2. The low dropout setting gives the best results on the test set and a high dropout setting gives the worst results - in happens because with high dropout the model has a high probability of dropping neurons which results in a model being inable to learn effectively and to underfit, while the model with a low dropout has a good balance between the regularization caused by dropout and learning capacity.
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible, for example:
During training, the model might produce slightly less confident predictions for the correct class (like probabilities shifting from 0.95 to 0.85 for the correct label). This could lead to an increase in loss because the cross-entropy loss penalizes lower confidence predictions, even if the model still predicts the correct class, which means, the accuracy remains the same or even increases while the loss increases.
"""

part2_q3 = r"""
**Your answer:**
1. Difference between Gradient Descent and Backpropagation:
Gradient Descent: It is an optimization algorithm that minimizes a loss function by always updating the parameters of the model in the direction of the negative gradient.
Backpropagation: The algorithm for computation of the gradient of the loss with respect to each model parameter using the chain rule. It is used internally to efficiently compute the gradients needed for gradient descent.
Key Difference:
Gradient descent is the optimization process of updating the parameters, while backpropagation is the method to compute the gradients for that process.

2. The comparison between gradient descent (GD) and stochastic gradient descent (SGD):
1 - Gradient computation:
    GD - uses the full dataset to compute gradient at each step
    SGD - uses a single data point or a small batch to compute the gradient
2 - Update Frequency:
    GD - updates parameters after processing the entire dataset.
    SGD - updates parameters after every data point (or batch).
3 - Speed:
    GD - slower due to full dataset computation.
    SGD - faster, because as it processes smaller portions of data.
4 - Memory Usage:
    GD - Requires the entire dataset to fit in memory.
    SGD - Requires only a single data point (or batch) in memory.
5 - Convergence:
    GD - Stable convergence but may get stuck in local minima.
    SGD - Faster convergence but noisier; can escape local minima.

3. SGD is used more, because it requires less memory, is faster and requires less computational power since it processes data in smaller batches, it can also be easily parallelized and is more suited for large datasets, that are hard or impossible to fit in memory.

4. A.  Yes, this approach would produce a gradient equivalent to GD:  
   Let the dataset consist of N samples: $(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)$. The gradient of the loss L w.r.t. the model parameters $\theta$ for GD is:  
   $
   \nabla L(\theta) = \frac{1}{N} \sum_{i=1}^{N} \nabla L_i(\theta)
   $
   If we split the data into M disjoint batches, compute the loss for each batch, and sum their gradients, the result will still be equivalent to computing the gradient over the entire dataset:
   $
   \nabla L(\theta) = \frac{1}{N} \sum_{j=1}^{M} \sum_{i \in \text{batch}_j} \nabla L_i(\theta)
   $
   Since each sample contributes its gradient exactly once, the gradients are mathematically equivalent to those of GD.
   
   B. The error might have occured, because we still accumulate the gradients and the loss for all the batches of a dataset, before we perform the backward pass, which means all the gradients and computations are stored for each batch, and this exceeds the available memory.


"""

part2_q4 = r"""
**Your answer:**
1. A. In forward mode AD, the memory complexity can be reduced by not storing all intermediate derivatives and function values, but instead, bycomputing them on-the-fly as we move forward through the chain of functions.
    Memory complexity: O(1), since we only need to store the current derivative and function value for each step.

    B. In backward mode AD, the memory complexity can be reduced by computing the necessary intermediate values during the backward pass rather than by storing them during the forward pass.
    Memory complexity: O(1), since we only need to store the current gradient during the backward pass and recompute intermediate values as needed.

2. Yes, these techniques can be generalized for arbitrary computational graphs.
In forward mode AD, we can simply propagate derivatives through each node in the graph, without storing all intermediate results. In backward mode AD, we can recompute the necessary values of the intermediates during the backward pass rather than store them, which  may involve a significant amount of additional computation, due to repeated evaluations, but requires less memory.

3. These memory-saving techniques can be very beneficial for backpropagation in deep models, because it can significantly reduce memory usage, which is critical for training deep networks with limited GPU memory, but increase computation time. This trade-off between computation and memory allows training larger models or using larger batch sizes.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

1) High optimization error occurs when the MLP struggles to minimize the training loss, leading to poor performance even on the training data. This may be caused by poor choice of optimizer or learning rate or vanishing/exploding gradients due to inappropriate weight initialization or activation functions. Thus we can reduce optimization error tuning learning rate, batch normalization or gradient clipping.
2) High deneralization error occurs due to overfitting. Thus, we can fix it by reducing overfitting, for example early stopping, regularization or simplifying the model(reducing its complexity).
3) High approximation error occurs due to underfiting, in other words, the model is to simple for the given problem. Thus we can fix it by increasing the complexity of the model, for example add more layers or use different activation function.
"""

part3_q2 = r"""
**Your answer:**

We may receive higher FPR in email spam detection, when good emails missclassified as spam. We may receive higher FNR in cancer detection, when a patient with cancer is missclassified as healthy.
"""

part3_q3 = r"""

**Your answer:**
In Scenario 1, where the disease leads to non-lethal symptoms enabling later diagnosis and treatment, the priority is minimizing the FPR to avoid unnecessary, costly, and risky tests, even at the cost of a higher FNR. Conversely, in Scenario 2, where missed early diagnosis can result in high mortality, the priority shifts to minimizing FNR to ensure early detection and prevent loss of life, even if this increases FPR. Thus, the "optimal" ROC point will differ: low FPR in Scenario 1 to reduce unnecessary tests and low FNR in Scenario 2 to minimize missed diagnoses.
"""


part3_q4 = r"""
**Your answer:**

There are a lot of reasons why MLP is not the best choise to work with sequntial data. It has fixed input size, but sequential data can be as long as possible, sequential data often relies on relationships between elements across time or position. For example, in a sentence, a word's sentiment can depend on preceding or following words. MLPs cannot inherently model such temporal dependencies, which are vital for sentiment classification.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**1) Regular block has 2 layers of 3x3 convolution, thus it has: 2x(3x3x256x64)=295040 parameters, when the bottelneck block has: (1x1x256x64)+64 + (3x3x64x64)+64 + (1x1x64x256)= 70106 paramters,, significantly fewer the regular block.
2) For regular block: first 3x3 convolution layer prerforms 3x3x256=2304 multiplications per pixel, thus HxWx64x2304=147456xHxW FLOPs total. Second 3x3 convolution layer performs 3x3x64=576 multiplications per pixel, thus  HxWx64x576=36864xHxW FLOPs total, for the whole block 184320xHxW FLOPs total. 
For bottleneck block: first 1x1 convolution layer performs 256 FLPOs per output element, thus HxWx64x245=16384xHxW FLOPs total. Second 3x3 convolution layer performs 3x3x64=674 FLOPs per output element, thus performs HxWx64x576=36864xHxW FLOPs total. Third 1x1 convolution layer performs 64 FLOPs per output element, thus thus HxWx64x245=16384xHxW FLOPs total. Totally leading to 69632xHxW FLOPs.
Thus regullar block performs approximately triple as much FLOPs as bottleneck block.
3)  1) Both blocks use 3x3 convolutions, which can capture local spatial patterns effectively, thus there is no significant difference in spatial combination.
    2) Regular block directly operates on 256 input channels in both 3x3 convolutions, allowing for richer cross-feature interactions, when Bottleneck block reduces the number of channels to 64 before processing, limiting the cross-feature map combination compared to the regular block.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

"""

part5_q2 = r"""
**Your answer:**


"""

part5_q3 = r"""
**Your answer:**


"""

part5_q4 = r"""
**Your answer:**



"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""