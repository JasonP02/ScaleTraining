Rotary positional embeddings is a modern transformer method for encoding positional information in a setnence.

[RoFormer Paper](https://arxiv.org/pdf/2104.09864)

* Encodes absolute position with a rotation matrix
* Incorporates relative position dependency in self-attention
Math
* Let $S_{n}$ be sequence of tokens with $n$ elements
* $E_{n}$ is the word embedding of $S$ with no positional information
* Before self attention, position is encoded by:

$$
q_{m}=f_{q}(x_{m,m})
$$
$$
k_{n}=f_{k}(x_{n},n)
$$
$$
v_{n}=f_{v}(x_{n},n)
$$
Paper's Approach
* Self attention allows knowledge between tokens to be shared
* To show relative position, the inner product of $q_{m},k_{n}$ is formulated by a function $g$ which takes word embeddings: 
$$
x_{m},x_{n}
$$
- ... and returns their relative position $(m-n)$ as inputs. The inner product, thus, encodes position as:
$$
f_{q}(x_{m},m),f_{k}(x_{n},n)=g(x_{m},x_{n},m-n)
$$
- ... thus, the goal is to find mechanisms $f_{q},f_{k}$ to create this mapping
- For $d=2$, the authors formulate this as:

$$
f_{q}(x_{m},m)=(W_{q}x_{m})e^{im\theta}
$$
$$
f_{k}(x_{n},n)=(W_{k}x_{n})e^{in\theta}
$$
![[Pasted image 20250816162826.png]]
where $i$ represents the index of the vectors

- For $x_{i}\in R^d$ , where $d$ is even, the $d$ dimensional space is split into $\frac{d}{2}$ sub-spaces which gives
$$
f_{q,k}(x_{m},m)=R^d_{\Theta,m}W_{q,k}x_{m}
$$

Where $R$ is a sparse rotation matrix with the form:

![[Pasted image 20250816163109.png]]

- $\Theta=(\theta_{i}=10000^{-2(i-1)/d})$ with $i\in[1,2,3,\dots,d/2]$
- The self attention equation is:
$$
q^T_{m}k_{n}=(R_{\Theta,m}W_{q}x_{m})^T(R_{\Theta,n}W_{k}x_{n}) =x^TW_{q}R_{\Theta,n-m}W_{k}x_{n}
$$
Where
$$
R_{\Theta,n-m}=(R_{\Theta,m})^T(R_{\Theta,n})
$$

My comments:
- It is clear that we first must initalize the rotation matrix. We know that $n,m$ are dependent on the sequence length. We can avoid looping by using matrix multiplicatiion
- But, I don't understand how to apply the equation for $q_{m}^Tk_{n}$ in the case of scaled_dot_product_attention, since this operation happens inside the function. In other words, the encoding of position has to happen beforehand
- Each section of $d_{hidden}$ essentially gets its own theta value
- We need to split up $x$ into $\frac{d}{2}$ dimensions for this pairwise operation...
- Okay, out of time.

Implementation
- Create $\Theta$ and all sin/cos pairs as seperate lists


I have been confused on this paper. I mixed up the required proving of the affine transformation properties with the implementation steps. In reality, the method allows for pairwise 2x2 matmuls which 