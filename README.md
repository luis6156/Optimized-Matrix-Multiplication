# Copyright Micu Florian-Luis 2022 - Assignment 2

# Purpose
This assignment proves various ways of optimizations for a computer program.
In particular, it is required that the following matrix equation is solved: 

C = B x A x A^t + B^t x B

There are three implementations for this problem: unoptimized, optimized (same
complexity as unoptimized) and BLAS.

# Implementation
## Unoptimized
I started creating the algorithm to solve the matrix equation by first creating
the transpose of the matrices and then I used the standard three-for-loops
algorithm for matrix multiplication. I stored the result in partial matrices
(B x A was stored in "partial_BA", partial_BA x A^t was stored in 
"partial_BAA_t" and B^t x B was stored in "partial_B_tB").

After I had the initial algorithm, I started to decrease the complexity of my
implementation by reducing the total number of for loops.

First, I computed the transpose of the matrices which I considered to not be
entirely necessary as I could use the already existing matrices and traverse
them differently. I taught that this solution might become inefficient as
for the case of B^t x B I had to iterate the first matrix column-wise, however
in the case of partial_BA x A^t I had to traverse the second matrix row-wise
which is more cache friendly as the processor prefetches the next elements and
for this case would be a direct cache hit. This problem will be addressed 
later.

Second, the matrix A is upper triangular therefore only half the normal 
computations are required which reduces a for loop to half. The same is
available for the transpose as it now becomes lower triangular. Since matrix
multiplication is comutative, I could start by multiplying first A x A^t and
then compute B x A x A^t, however this would not yield any significant 
performance increase as A x A^t will become a matrix that is not triangular,
thus B x (A x A^t) will have the same number of computations as the 
(B x A) x A^t.

Third, a lot of computations could be done in paralel, for example B x A and
B^t x B could be computed at the same time. Moreover, some matrices do not
need to be fully computed to be used in further multiplications, hence I 
reduced drastically the total number of for loops.

The total complexity for the algorithm is O(N^3), since the matrices are 
square.

No other optimizations where done, as that would be the task for the optimized
version.  

### Note
Stack allocation is faster than the heap allocation, however we need to 
allocate many N x N matrices, where N <= 1600. This is not feasiblei, as the
stack space is smaller, thus we are required to use the heap and free the
memory afterwards. 

## Optimized
Based on the unoptimized version, I started optimizing by making sure I do not
compute things more than once and that I assure more cache hits.

First, matrices are held in memory as vectors which is already efficient, 
however we could further optimize this by not computing the indices twice, 
hence I saved the index in a variable. Even more, it still not be fully
optimal as the CPU gets the address of the vector and adds the index we
computed earlier. Instead, I used a pointer to the vector that holds its
address and I incremented it everytime its associated for loop ends (some 
pointers could not be incremented as they required to have a constant value
and jump suddenly N positions, therefore I added to the pointer N). 
Incrementing is faster than addition as it required only one clock cycle,
furthermore I used prefixed incrementation as it is faster than its postfix
variant.

Second, I changed the order of the for loops so that I can get more cache hits.
In the standard form (i-j-k), we get a constant memory write access C[i][j], a
sequential memory read access A[i][k] and a nonsequential memory read access 
B[k][j] for the following formula C[i][j] = A[i][k] * B[k][j]. The 
nonsequential part hurts our performance as the CPU prefetches the next elements
row-wise (since this is how we store our matrices) and this results in a cache 
miss everytime, thus the CPU has to fetch data with more latency. Instead if we
use the i-k-j loops, we get sequential memory write access, constant memory 
read access and sequential memory write access. This is better as it results
in more cache hits (cache misses will be still found as the cache memory is 
finite, however they will be significantly less frequent). 

Third, I used the "register" directive to signal to the assembler to store the
variables in the registers as much as possible since they are the fastest type
of memory.

### Note
1. Two for loops were interchanged as the A matrix is triangular and the k loop
had to start from j in the initial form, therefore for the i-k-j form the j 
loop has to start from k. The complexity is exactly the same and there are no
advantages or disadvantages to this switch. Therefore, the total complexity
remained O(N^3). Look in the code for a clearer picture.
2. I Tried to do SSE optimizations as the program could compute more values
at once, however we were not taught how to use them and I had a hard time
understanding them. This would have been a major boost in performance as the
BLAS version uses SSE optimizations heavily. 

## BLAS
BLAS is a library used by data scientist specifically for very fast 
computations. This library has different "levels" for different types of 
computations (Level 1 - simple element equations / Level 2 - vector equations /
Level 3 - matrix equations). Each level also has multiple versions for each
function where only the accuracy of the data is different (float, double etc.).
I used the cblas_dgemm function as it represents the equation C = �AB + �C for
standard matrices A, B and C and the cblas_dtrmm function as it represents the
equation B = �AB where B is a rectangular matrix and A is a triangular matrix.

Legend:

cblas_dtrmm(cblas_layout, cblas_side, cblas_uplo, cblas_transa, cblas_diag, m,
						n, alpha, a, lda, b, ldb)

layout -> indicates whether the input and output matrices are stored in row 
					major order or column major order
side -> indicates whether the triangular matrix A is located to the left or 
			right of rectangular matrix B in the equation used
	if left -> A is to the left of B in the equation
	else -> A is to the right of B in the equation
uplo -> indicates whether matrix A is an upper or lower triangular matrix
transa -> indicates whether matrix A needs to be transposed
diag -> indicates whether matrix A is a unit matrix
m -> number of rows in rectangular matrix B
n -> number of columns in rectangular matrix B
alpha -> scalar
a -> matrix A
lda -> leading dimension of the array specified for a
b -> matrix B
ldb -> leading dimension of the array specified for b

cblas_dgemm(cblas_layout, cblas_transa, cblas_transb, l, n, m, alpha, a, lda,
						b, ldb, beta, c, ldc)

transa/transb -> indicate wether the matrix A/B will be transposed or not
l -> number of rows in matrix C
n -> number of columns in matrix C
m -> if transa is true -> number of rows in A; else -> number of columns in A
alpha -> scalar
a -> matrix A
lda -> leading dimension of the array specified for a
b -> matrix B
ldb -> leading dimension of the array specified for b
beta -> scalar
c -> matrix C
ldc -> leading dimension of the array specified for c

### Computing B x A
Supporting function needs to have support for triangular matrix since A is
upper triangular, thus I used cblas_dtrmm with the following parameters:

layout -> CblasRowMajor (row-major)
side -> CblasRight (triangular matrix is to the right)
uplo -> CblasUpper (upper triangular matrix)
transa -> CblasNoTrans (triangular matrix does not need to be transposed)
diag -> CblasNonUnit (triangular matrix is not unit matrix)
m -> N
n -> N
alpha -> 1
a -> matrix A
lda -> N
b -> matrix partial_C
ldb -> N

The result will be stored in the matrix "partial_C".

#### Note
The matrix "partial_C" is a copy of the matrix B. I needed a copy since the
cblas_dtrmm function overrides the matrix B to store the result and I will
need that particular matrix again for computing B^t x B.

### Computing B x A x A^t (partial_c x A^t)
Supporting function needs to have support for triangular matrix since A^t is
lower triangular, thus I used cblas_dtrmm with the following parameters:

layout -> CblasRowMajor (row-major)
side -> CblasRight (triangular matrix is to the right)
uplo -> CblasUpper (upper triangular matrix)
transa -> CblasTrans (triangular matrix does need to be transposed)
diag -> CblasNonUnit (triangular matrix is not unit matrix)
m -> N
n -> N
alpha -> 1
a -> matrix A
lda -> N
b -> matrix partial_C
ldb -> N

The result will be stored in the matrix "partial_C".
 
### Computing B^t x B & the final result
Supporting function needs to have support for multipling two normal matrices
and for an addition after the product has been done, thus I used cblas_dgemm
with the following parameters:

layout -> CblasRowMajor (row-major)
transa -> CblasTrans (matrix A does need to be transposed)
transb -> CblasNoTrans (matrix B does not need to be transposed)
l -> N
n -> N
m -> N
alpha -> 1
a -> matrix B
lda -> N
b -> matrix B
ldb -> N
beta -> 1
c -> matrix partial_C
ldc -> N

The result will be stored in partial_C.

#### Note
In this step I compute B^t x B at the same time with the addition 
B x A x A^t + B^t x B. Since matrix addition is associative, I could do 
B^t x B + B x A x A^t and it would still yield the correct result. 

# Valgrind Analysis
## Memcheck
### Optimized and unoptimized
As stated by valgrind, all of the memory has been freed and only 10 allocs have
been done. The partial matrices used in my algorithm have been freed immediatly
whilst the matrix returned will be freed later.

### BLAST
The blas library apparently uses more allocation, 79 to be precise, however all
of the memory is freed and as we will see it remains the fastest algorithm. 

## Cachegrind
## How it works
When running cachegrind, it is important to understand the result returned.
According to the documentation, the most relevant levels for caching are the
first and the last. Usually, cachegrind simulates L1 and L2 caches, however if 
it detects that the CPU has more levels, it will only simulate the L1 and 
last-level cache. Regarding the output itself, the first section represents 
the cache analysis for the instructions for L1 and LL cache (shows if the 
instructions managed to fit in the cache). The second section represents the L1
and LL data cache analysis (usually the sought after part as it dictates the 
performance of the algorithm). The last part represents the performance of the
LL cache for both instructions and data.

Legend:
I refs - total instructions
I1 misses - instruction cache L1 misses
LLi - instruction cache LL misses
I1 miss rate - instruction percentage miss rate for L1 cache
LLi miss rate - instruction percentage miss rate for LL cache
D refs - total data
rd - read
wr - write
D1 misses - data misses for L1 cache
LLd misses - data misses for LL cache
D1 miss rate - data percentage miss rate for L1 cache
LLd miss rate - data percentage miss rate for LLi cache
LL refs - total memory used in LL cache (data + instructions)
LL misses - total cache misses for LL cache
LL miss rate - total percentage miss rate for LL cache

Moreover, the branches were simulated by valgrind indicating how performant was
the branch predictor and it also shows many branches the program produced. 
Conditional jumps are dictated by if clauses, whilst indirect branches are 
caused by jumps to unknown destinations (e.g. jumping to a memory address).

Legend:
branches - total branches
cond - conditional branches
ind - indirect branches
mispredicts - how many times the branch predictor was wrong
mispred rate - percentage of wrong branch prediction assumptions

## Analysis
### Instructions
			| BLAS			| OPTIMIZED		| UNOPTIMIZED
Total Instructions	| 247,837,210		| 1,568,111,313		| 5,795,464,307
I1 misses		| 16,459		| 1,616			| 1,620
LLi misses		| 3,199			| 1,540			| 1,541

Percentages were omitted as they are approximately 0.0% for all of the 
versions regardless of the cache.

From the table, it is clear that instruction misses are not a bottleneck,
however it is interesting to note that the BLAS version generates at least
x5 times less instructions. In addition, the optimized variant did not 
represent a major change in terms of misses, however it generated less 
instructions as this variant uses pointers instead of classic pointers to 
access vectors which translates to less instruction references (addresses are
already computed, thus the CPU does not have to do multiplications and 
additions and it falls on only incrementing a pointer which is faster).

### Data
                        | BLAS                  | OPTIMIZED             | UNOPTIMIZED
Total Data      	| 92,369,137         	| 814,021,530         	| 2,865,698,894
D1 misses               | 1,607,951             | 16,610,396            | 166,263,179
LLd misses              | 97,009                | 113,268               | 113,266
D1 miss rate		| 1.7%			| 2.0%			| 5.8%
LLd miss rate		| 0.0%			| 0.0%			| 0.0%

From the table, the total data stored in the BLAS version is almost 9x times 
less than the optimized version and 31x times less than the unoptimized 
version. The main bottleneck seems the L1 cache data misses, since for the
BLAS version there are 10x times less misses than the optimized version and 
almost 100x times less misses than the unoptimized version. It is interesting
to notice that the LL cache misses are not that different, however if we take
in consideration the total data stored, BLAS has a cache miss rate bigger than
the optimized and unoptimized versions. According to a StackOverflow answer, a
huge number of L1 misses and a small number of LL misses might be a sign that,
in the context of matrix multiplications, data is accessed column wise which
negates the benefits of the prefetching. This scenario is true as data is 
accessed in the worst case sequentially which means there will be a lot more
cache hits for the optimized version (i-k-j optimization).

#### Note
From the cachegrind output, the optimized version generates far less read 
misses and data as the i-k-j optimization changed the last operand of an 
equation to become sequential. In the context of matrix multiplications,
the last operand represents a read operation which aligns with our remark. 

### Branches
                        | BLAS                  | OPTIMIZED             | UNOPTIMIZED
Total Branches      	| 4,423,388           	| 99,829,279         	| 99,829,187
Mispredicts             | 67,714                | 503,560               | 503,535
Mispredicts rate        | 1.5%                  | 0.5%                  | 0.5%

The total number of branches generated by the BLAS version is 22x times less
than the optimized and unoptimized versions. It is interesting to notice that
the optimized and unoptimized versions are extremely similar and the optimized
version has more mispredicts than the unoptimized one. The percentages are not
that good for analysis as they are associated to the total number of branches
and altough it appears that the rate of mispredicts is bigger for the BLAS
version, it is only partially true, as if we look at the total number of
mispredicts we can clearly see that there are 10x times less mispredicts in
the BLAS version which would translate into less hustle from the CPU.

#### Note for general performance
Altough it is hard to analyse from the cachegrind output, the "register" 
keyword helped the performance as it would translate to faster memory access.

# Graphs & Performance
The standard tests are only mentioning N = 400, 800, 1200, therefore I added
measurements for N = 1000 and 1400. According to the graph, the unoptimized
version has an exponential curve, indicating that with a large value for N
the performance will drop dramatically. For the optimized version, we can see
see that its performance is constant, just like the BLAS version, however the
slope for the optimized version is bigger indicating that BLAS is the best
version.

Furthermore, analysing the values themselves, it is clear that BLAS has an
advantage. For N = 1400, BLAS runs in 0.91s, the optimized version runs in
14.04s and the unoptimized version in 60.75s, therefore BLAS runs 15x/66x 
times faster than the rest. Even for small values of N such as 400, BLAS
runs in 0.03s, the optimized version runs in 0.31s and the unoptimized
version runs in 1.18s which means that BLAS runs 10x/40x times faster.

## Note
1. The optimized algorithm managed to run in 8.75s consistently for N = 1200.    
2. The times were checked by running the tests 3 times on each version on
nehalem.

# Bibliography
https://ocw.cs.pub.ro/courses/asc/laboratoare/05
https://stackoverflow.com/questions/20172216/how-do-you-interpret-cachegrind-output-for-caching-misses
https://www.ibm.com/docs/en/essl/6.3?topic=mos-sgemm-dgemm-cgemm-zgemm-combined-matrix-multiplication-addition-general-matrices-their-transposes-conjugate-transposes
https://www.ibm.com/docs/en/essl/6.1?topic=mos-strmm-dtrmm-ctrmm-ztrmm-triangular-matrix-matrix-product

Sorry for the long README... :(
