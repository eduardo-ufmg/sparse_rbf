### Overview

The procedure computes a sparse multivariate Gaussian Radial Basis Function (RBF) kernel matrix. Given two sets of points, $X$ and $Y$, in a multi-dimensional space, the goal is to construct a matrix $K$ where each entry $K_{ij}$ represents the kernel similarity between the $i$-th point in $X$ and the $j$-th point in $Y$.

The key feature is **sparsity**: instead of computing the similarity for all possible pairs of points, it only calculates it for the $k$ closest points in $Y$ for each point in $X$, setting all other values to zero. This is highly efficient for large datasets.

### Mathematical Formulation

Let $X = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$ and $Y = \{\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_m\}$ be two sets of vectors, where each $\mathbf{x}_i, \mathbf{y}_j \in \mathbb{R}^d$ for some dimension $d$.

The computation requires two parameters:
1.  A bandwidth parameter $h \in \mathbb{R}^+$.
2.  The number of nearest neighbors $k \in \mathbb{Z}^+$.

#### 1. Gaussian RBF Kernel

The Gaussian RBF kernel is a function that measures the similarity between two vectors, $\mathbf{u}$ and $\mathbf{v}$. It is defined as:

$$K(\mathbf{u}, \mathbf{v}) = \exp\left(-\frac{\|\mathbf{u} - \mathbf{v}\|^2}{2h^2}\right)$$

Here, $\|\mathbf{u} - \mathbf{v}\|$ denotes the Euclidean distance between the two vectors. The value of the kernel is always in the range $(0, 1]$, reaching its maximum of 1 when the vectors are identical. The bandwidth $h$ controls the width of the kernel; a smaller $h$ leads to a more localized and sensitive similarity measure.

#### 2. Sparsification with K-Nearest Neighbors

To create a sparse matrix, we first identify a neighborhood for each point $\mathbf{x}_i \in X$. For each $\mathbf{x}_i$, we find the set of its $k$ nearest neighbors within the set $Y$. This neighborhood, denoted as $\mathcal{N}_k(\mathbf{x}_i)$, is formally defined as:

$$\mathcal{N}_k(\mathbf{x}_i) \subseteq Y \quad \text{such that} \quad |\mathcal{N}_k(\mathbf{x}_i)| = k$$

and for any $\mathbf{y}_a \in \mathcal{N}_k(\mathbf{x}_i)$ and any $\mathbf{y}_b \in Y \setminus \mathcal{N}_k(\mathbf{x}_i)$, the following condition holds:

$$\|\mathbf{x}_i - \mathbf{y}_a\| \le \|\mathbf{x}_i - \mathbf{y}_b\|$$

This means that the set $\mathcal{N}_k(\mathbf{x}_i)$ contains the $k$ points from $Y$ that are closest to $\mathbf{x}_i$ based on Euclidean distance.

#### 3. Sparse Kernel Matrix Construction

The final output is an $n \times m$ sparse matrix $K$. Each element $K_{ij}$ of this matrix, corresponding to the pair $(\mathbf{x}_i, \mathbf{y}_j)$, is defined as follows:

$$
K_{ij} =
\begin{cases}
\exp\left(-\frac{\|\mathbf{x}_i - \mathbf{y}_j\|^2}{2h^2}\right) & \text{if } \mathbf{y}_j \in \mathcal{N}_k(\mathbf{x}_i) \\
0 & \text{otherwise}
\end{cases}
$$

By this construction, each row $i$ of the matrix $K$ will contain exactly $k$ non-zero elements, corresponding to the kernel values computed for the $k$-nearest neighbors of $\mathbf{x}_i$ in $Y$. All other entries in the row are zero, resulting in a sparse and memory-efficient matrix representation of the kernel similarities.