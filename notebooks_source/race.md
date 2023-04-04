---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# New JIT and AOT Compilers

+++

[John Stachurski](http://johnstachurski.net)

```{code-cell} ipython3
!pip install quantecon
```

```{code-cell} ipython3
import numpy as np
from numba import vectorize, jit, float64
from quantecon.util import tic, toc
import matplotlib.pyplot as plt
```

## Problem 1: A Time Series Model

+++

Consider the time series model

$$ x_{t+1} = \alpha x_t (1 - x_t) $$

Let's set $\alpha = 4$

```{code-cell} ipython3
α = 4
```

Here's a typical time series:

```{code-cell} ipython3
n = 200
x =  np.empty(n)
x[0] = 0.2
for t in range(n-1):
    x[t+1] = α * x[t] * (1 - x[t])
    
plt.plot(x)
plt.show()
```

## Python Test

+++

Here's a function that simulates for `n` periods, starting from `x0`, and returns **only the final** value:

```{code-cell} ipython3
def quad(x0, n):
    x = x0
    for i in range(1, n):
        x = α * x * (1 - x)
    return x
```

Let's see how fast this runs:

```{code-cell} ipython3
n = 10_000_000
```

```{code-cell} ipython3
tic()
x = quad(0.2, n)
toc()
```

## Fortran Test

+++

Now let's try this in Fortran.

Note --- this step is intended to be a demo and will only execute if

* you have the file `fastquad.f90` in your pwd
* you have a Fortran compiler installed and modify the compilation code below appropriately

```{code-cell} ipython3
%%file fortran_quad.f90

PURE FUNCTION QUAD(X0, N)
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                           
 REAL(dp), INTENT(IN) :: X0
 REAL(dp) :: QUAD
 INTEGER :: I
 INTEGER, INTENT(IN) :: N
 QUAD = X0
 DO I = 1, N - 1                                                
  QUAD = 4.0_dp * QUAD * real(1.0_dp - QUAD, dp)
 END DO
 RETURN
END FUNCTION QUAD

PROGRAM MAIN
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                          
 REAL(dp) :: START, FINISH, X, QUAD
 INTEGER :: N
 N = 10000000
 X = QUAD(0.2_dp, 10)
 CALL CPU_TIME(START)
 X = QUAD(0.2_dp, N)
 CALL CPU_TIME(FINISH)
 PRINT *,'last val = ', X
 PRINT *,'elapsed time = ', FINISH-START
END PROGRAM MAIN
```

```{code-cell} ipython3
!gfortran -O3 fortran_quad.f90
```

```{code-cell} ipython3
!./a.out
```

```{code-cell} ipython3
!rm a.out
```

## Python + Numba

+++

Now let's do the same thing in Python using Numba's JIT compilation:

```{code-cell} ipython3
quad_jitted = jit(quad)
```

```{code-cell} ipython3
tic()
x = quad_jitted(0.2, n)
toc()
```

```{code-cell} ipython3
tic()
x = quad_jitted(0.2, n)
toc()
```


## Codon


```{code-cell} ipython3
!/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

```{code-cell} ipython3
%%file codon_quad.py

from time import time

n = 10_000_000
alpha = 4.0

def quad(x0, n):
    x = x0
    for i in range(1, n):
        x = alpha * x * (1 - x)
    return x


t0 = time()
x = quad(0.1, n)
t1 = time()
print(x)
print(t1 - t0)
```

```{code-cell} ipython3
!codon build --release --exe codon_quad.py
```

```{code-cell} ipython3
!./codon_quad
```

```{code-cell} ipython3
!rm codon_quad
```

```{code-cell} ipython3

```
