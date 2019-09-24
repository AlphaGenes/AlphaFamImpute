from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import BasicHMM
from ..tinyhouse import ProbMath
from ..tinyhouse import InputOutput
from . import FamilySingleLocusPeeling
import math
import random
import numpy as np

from numba import jit, njit
import math

def imputeFamUsingFullSibs(fam, pedigree, args) :

    #Pipeline:
    # 0) Get LD and HD children.
    # 1) Take all the children and phase via parents homozygous loci.
    # 2) Take all of the LD children and impute from the parents.

    # STEP 0: Get LD and HD children.
    nLoci = len(fam.sire.genotypes)
    ldChildren = fam.offspring
    hdChildren = fam.offspring

    runImputationRound(fam, ldChildren, hdChildren)

def runImputationRound(fam, ldChildren, hdChildren, callingMethod = "dosages", preimpute = False) :
    #STEP 1: Take all of the HD children and phase/(impute?) the parents.

    founderImputation(fam.sire, fam.dam, hdChildren, preimpute = preimpute)



def founderImputation(sire, dam, children, preimpute = False):
    
    nChildren = len(children)
    nLoci = len(sire.genotypes)
    ref_reads = np.full((nChildren, nLoci), 0, dtype = np.float32)
    alt_reads = np.full((nChildren, nLoci), 0, dtype = np.float32)

    for i in range(nChildren):
        ref_reads[i,:] = children[i].reads[0]
        alt_reads[i,:] = children[i].reads[1]

    # The forward pass works in a couple of steps.
    # Estimate p(y|h,x) for each loci.
    # Use that estimate to calculate the parent estimate. g(h)
    # Normalize the "a" estimate across parental haplotypes.

    # Some notation. Each parent has four possible haplotypes, aa, aA, Aa, AA.
    # The child as four inheritance patterns, mm, mp, pm, pp.
    # We may be able to re-use some of the segregation archetecture.

    child_point_estimates = np.full((nChildren, 4, 4, 4, nLoci), 0, dtype = np.float32) # Three dimensional array for each individual. First index is segregation, then father and mother's haplotypes.
    
    for child in range(nChildren):
        fill_child_point_estimate(child_point_estimates[child,:,:,:,:], ref_reads[child], alt_reads[child])

    parent_point_estimates = np.full((4, 4, nLoci), 0, dtype = np.float32) # Sire and dam. For their four haplotype states.
    fill_parent_point_estimate(parent_point_estimates, sire.reads[0], sire.reads[1], dam.reads[0], dam.reads[1])

    parent_haplotypes, child_haplotypes = perform_updates(child_point_estimates, parent_point_estimates)

    sire.genotypes[:] =   parent_haplotypes[0,0,:] + parent_haplotypes[0,1,:] 
    sire.haplotypes[0][:] = parent_haplotypes[0,0,:]
    sire.haplotypes[1][:] = parent_haplotypes[0,1,:]


    dam.genotypes[:] =   parent_haplotypes[1,0,:] + parent_haplotypes[1,1,:] 
    dam.haplotypes[0][:] = parent_haplotypes[1,0,:]
    dam.haplotypes[1][:] = parent_haplotypes[1,1,:]

    # for i in range(nChildren):
    #     children[i].genotypes[:] = child_haplotypes[i,0,:] + child_haplotypes[i,1,:] 
    #     children[i].haplotypes[0][:] = child_haplotypes[i,0,:]
    #     children[i].haplotypes[1][:] = child_haplotypes[i,1,:]

    # print(parent_haplotypes[0,:,0:10])
    # print(parent_haplotypes[1,:,0:10])

    for child in children:
        BasicHMM.diploidHMM(child, parent_haplotypes[0,:,:], parent_haplotypes[1,:,:], 0.01, 1.0/nLoci, useCalledHaps = False, callingMethod = "dosages")


# @njit
def perform_updates(child_point_estimates, parent_point_estimates):
    nChildren = child_point_estimates.shape[0]
    nLoci = child_point_estimates.shape[4]

    initial_seg = np.random.randint(0, 4, size = nChildren)
    parent_genotypes, child_seg = maximization_pass(parent_point_estimates, child_point_estimates, initial_seg, False)
    np.savetxt("outputs/initial_seg.txt", child_seg, fmt="%i")

    # parent_estimate, aNorm = forward_pass(parent_point_estimates, child_point_estimates, parent_genotypes, fixed_loci = np.repeat([1, 0], nLoci/2), zeroToN = True) # Breaks for even number of loci.

    # parent_genotypes, child_seg = maximization_pass(parent_estimate, aNorm, initial_seg, False)
    # np.savetxt("outputs/second_seg.txt", child_seg, fmt="%i")


    parent_haplotypes = extract_parent_haplotypes(parent_genotypes)
    child_haplotypes = extract_child_haplotypes(child_seg, parent_genotypes)


    return parent_haplotypes, child_haplotypes

# @njit
# def forward_pass(parent_point_estimates, child_point_estimates, parent_genotypes, fixed_loci, zeroToN) :
#     nChildren = child_point_estimates.shape[0]
#     nLoci = child_point_estimates.shape[-1]

#     if zeroToN:
#         start = 0
#         stop = nLoci
#         step = 1
#     else:
#         start = nLoci - 1
#         stop = -1
#         step = -1

#     parent_estimate = np.full((4, 4, nLoci), 0, dtype = np.float32) # Haplotypes. One dimension for each parent. This is not going to scale, but maybe some independence statements?

#     aTilde = np.full(child_point_estimates.shape, 0, dtype = np.float32)
#     aNorm = np.full(child_point_estimates.shape, 0, dtype = np.float32)

#     first_loci = True

#     forward_seg = np.full((nChildren, 4, nLoci), 0, dtype = np.float32)
#     for i in range(start, stop, step):
#         for child in range(nChildren):
#             if first_loci:
#                 forward = np.zeros(4, dtype = np.float32)
#             else:
#                 if fixed_loci[i]:
#                     genotype_matix = get_genotype_matrix(parent_genotypes[0, i-step], parent_genotypes[1, i-step])
#                     forward = combined_and_transmit(aNorm[child,:,:,:,i-step], genotype_matix)
#                 else:
#                     forward = combined_and_transmit(aNorm[child,:,:,:,i-step], parent_estimate[:, :, i-step])

#             # Loop is for annoying numpy notation to make sure the right axes get added.
#             for sire in range(4):
#                 for dam in range(4):
#                     aTilde[child, :, sire, dam,i] = child_point_estimates[child,:, sire, dam, i] + forward[:]

#         if first_loci:
#             first_loci = False

#         parent_estimate[:,:,i] = parent_point_estimates[:,:,i]
#         for child in range(nChildren):
#             parent_estimate[:,:,i] += log_sum_seg(aTilde[child, :, :, :, i])


#         for child in range(nChildren):
#             # Loop is for annoying numpy notation to make sure the right axes get added.
#             log_sum = log_sum_seg(aTilde[child, :, :, :, i])
#             for seg in range(4):
#                 aNorm[child, seg, :,:, i] = aTilde[child, seg, :, :, i] - log_sum[:,:]

            
#             if i == 0 and child == 1:
#                 print("aTilde", i, child)
#                 for seg in range(4):
#                     print(aTilde[child, seg, :,:, i])

#                 print("aNorm", i, child)
#                 for seg in range(4):
#                     print(aNorm[child, seg, :,:, i])

#     return parent_estimate, aNorm

@njit
def maximization_pass(parent_estimate, child_estimate, initial_seg, zeroToN):
    nChildren = child_estimate.shape[0]
    nLoci = child_estimate.shape[-1]

    parent_genotypes = np.full((2, nLoci), 9, dtype = np.int64) 
    beta = np.full(child_estimate.shape, 0, dtype = np.float32)

    # Doing first index.

    sire_score = np.full((4, 4), 0, dtype = np.float32)

    if zeroToN:
        start = 0
        stop = nLoci
        step = 1
    else:
        start = nLoci - 1
        stop = 990
        step = -1

    first_loci = True
    child_seg = np.full((nChildren, nLoci), 9, dtype = np.int64)

    for i in range(start, stop, step):

        # Set beta for each child.
        projected_seg = np.full((nChildren, 4), .25, dtype = np.float32)

        for child in range(nChildren):
            if first_loci:
                projected_seg[child,:] = .25
                tmp_seg = initial_seg[child]
            else:
                tmp_seg = child_seg[child, i - step]
                projected_seg[child,:] = get_transmitted_seg_matrix(tmp_seg)
            # add_backwards_sample(child_estimate[child,:,:,:,i], tmp_seg, beta[child, :, :, :, i]) # i+1 since we are including infromation from the loci we just sampled. 
 
        if first_loci:
            first_loci = False
        # calculate Ht for the parents.

        sire_score[:,:] = parent_estimate[:, :, i]
        for child in range(nChildren):
            sire_score += log_sum_seg(child_estimate[child, :, :, :, i], projected_seg[child,:])

        unraveled_index = max_multisample(sire_score)  #At last loci.
        sire_geno, dam_geno = numba_unravel(unraveled_index, parent_estimate[:,:,i].shape)
        
        print(i, unraveled_index, sire_geno, dam_geno)
        print(parent_estimate[:, :, i])
        print(sire_score)

        parent_genotypes[0, i] = sire_geno
        parent_genotypes[1, i] = dam_geno

        # calculate xt for each child.

        for child in range(nChildren):
            probs = child_estimate[child, :, sire_geno, dam_geno, i] # + np.log(projected_seg[child,:]) # The log projected_seg gives some prior based on previous loci.
            child_seg[child,i] = max_multisample(probs)
            # child_seg[child,i] = max_multisample(beta[child, :, sire_geno, dam_geno, i])

    return parent_genotypes, child_seg

@njit
def extract_parent_haplotypes(parent_genotypes):
    nLoci = parent_genotypes.shape[-1]
    parent_haplotypes = np.full((2, 2, nLoci), 9, dtype = np.int8)
    for parent in range(2):
        for i in range(nLoci):
            parent_haplotypes[parent, 0, i], parent_haplotypes[parent, 1, i] = parse_genotype(parent_genotypes[parent, i])

    return parent_haplotypes

@njit
def extract_child_haplotypes(child_seg, parent_genotypes):
    nChildren = child_seg.shape[0]
    nLoci = child_seg.shape[-1]
    child_haplotypes = np.full((nChildren, 2, nLoci), 9, dtype = np.int8)

    for child in range(nChildren):
        extract_haplotype(child_seg[child,:], parent_genotypes, child_haplotypes[child, :, :] )

    return child_haplotypes
@njit
def extract_haplotype(child_seg, parent_genotypes, output):
    nLoci = child_seg.shape[0]
    for i in range(nLoci):
        seg = child_seg[i]
        sire_geno = parent_genotypes[0, i]
        dam_geno = parent_genotypes[1, i]
        
        seg_sire, seg_dam = parse_segregation(seg)

        output[0, i] = parse_genotype(sire_geno)[seg_sire]
        output[1, i] = parse_genotype(dam_geno)[seg_dam]


@njit
def get_genotype_matrix(sire_genotype, dam_genotype):
    mat = np.zeros((4, 4), dtype = np.float32)
    mat[sire_genotype, dam_genotype] = 1
    return mat


@njit
def numba_unravel(index, shape):
    nx, ny = shape

    x = index // ny
    y = index % ny
    return(x,y)
@njit
def add_backwards_sample(aNorm, seg, beta):

    seg_matrix = np.log(get_transmitted_seg_matrix(seg))
    for sire in range(4):
        for dam in range(4):
            beta[:, sire, dam] = aNorm[:,sire, dam] + seg_matrix


@njit
def combined_and_transmit(child_estimate, parent_estimate):

    log_product = child_estimate + parent_estimate
    prev = np.full(4, 0, dtype = np.float32)
    output = np.full(4, 0, dtype = np.float32)

    max_value = np.max(log_product)

    for i in range(4):
        prev[i] = np.sum(np.exp(log_product[i, :, :] - max_value))

    prev = prev/np.sum(prev)

    return(transmit(prev, output))


@njit
def get_transmitted_seg_matrix(seg):

    prev = np.full(4, 0, dtype = np.float32)
    output = np.full(4, 0, dtype = np.float32)

    prev[seg] = 1
    return(transmit(prev, output))


@njit
def transmit(vect, output):

    e = .001
    e2 = e**2
    e1e = e*(1-e)
    e2i = (1.0-e)**2

    output[0] = e2*vect[3] + e1e*(vect[1] + vect[2]) + e2i*vect[0] 
    output[1] = e2*vect[2] + e1e*(vect[0] + vect[3]) + e2i*vect[1] 
    output[2] = e2*vect[1] + e1e*(vect[0] + vect[3]) + e2i*vect[2] 
    output[3] = e2*vect[0] + e1e*(vect[1] + vect[2]) + e2i*vect[3] 

    return output


@njit
def log_sum_seg(matrix, weights) :
    output = np.full((4,4), 0, dtype = np.float32)
    for pat in range(4):
        for mat in range(4):
            # print(matrix[:, pat, mat], log_sum_exp_1D(matrix[:, pat, mat]))
            output[pat, mat] = log_sum_exp_1D(matrix[:, pat, mat], weights) # Residual on parent genotypes.
    return output


@jit(nopython=True, nogil = True)
def log_sum_exp_1D(mat, weights):
    log_exp_sum = 0
    first = True
    maxVal = 100
    for a in range(4):
        if mat[a] > maxVal or first:
            maxVal = mat[a]
        if first:
            first = False

    for a in range(4):
        log_exp_sum += weights[a]*np.exp(mat[a] - maxVal)
    
    return np.log(log_exp_sum) + maxVal


# @jit(nopython=True, nogil = True)
# def log_sum_exp_2D(mat):
#     log_exp_sum = 0
#     max_val = 1
#     for a in range(4):
#         for b in range(4):
#             if mat[a, b] > maxVal or maxVal == 1:
#                 maxVal = mat[a, b]

#     # Should flag for better numba-ness.
#     for a in range(4):
#         for b in range(4):
#             log_exp_sum += np.exp(mat[a, b] - maxVal)

#     return np.log(log_exp_sum) + maxVal



@jit(nopython=True, nogil = True)
def exp_1D_norm(mat):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(mat.shape[0]):
        if mat[a] > maxVal or maxVal == 1:
            maxVal = mat[a]
    
    # Should flag for better numba-ness.
    tmp = np.full(mat.shape[0], 0, dtype = np.float32)
    for a in range(mat.shape[0]):
        tmp[a] = np.exp(mat[a] - maxVal)

    norm_1D(tmp)

    return tmp




@njit
def fill_child_point_estimate(point_estimate, ref_reads, alt_reads) :

    for seg in range(4):
        for sire in range(4):
            for dam in range(4):
                genotype = get_genotype(seg, sire, dam)[2]
                point_estimate[seg, sire, dam,:] = get_genotype_probs(genotype, ref_reads, alt_reads)


@njit
def fill_parent_point_estimate(point_estimate, sire_ref, sire_alt, dam_ref, dam_alt) :
    for sire in range(4):
        sire_0, sire_1 = parse_genotype(sire)
        sire_geno = sire_0 + sire_1

        for dam in range(4):
            dam_0, dam_1 = parse_genotype(dam)
            dam_geno = dam_0 + dam_1

            point_estimate[sire, dam,:] = get_genotype_probs(sire_geno, sire_ref, sire_alt)
            point_estimate[sire, dam,:] += get_genotype_probs(dam_geno, dam_ref, dam_alt)



@njit
def get_genotype_probs(genotype, ref_reads, alt_reads):
    loge = np.log(0.001)
    log1e = np.log(1-0.001)
    log2 = np.log(.5)

    if genotype == 0: values = ref_reads*log1e + alt_reads*loge
    if genotype == 1: values = log2*(ref_reads + alt_reads)
    if genotype == 2: values = alt_reads*log1e + ref_reads*loge

    return values.astype(np.float32)
@njit
def get_genotype(seg, sire, dam):

    seg_sire, seg_dam = parse_segregation(seg)

    geno_sire = parse_genotype(sire)[seg_sire]
    geno_dam = parse_genotype(dam)[seg_dam]

    return geno_sire, geno_dam, geno_sire + geno_dam

@njit
def parse_genotype(geno):
    if geno == 0: return (0, 0)
    if geno == 1: return (0, 1)
    if geno == 2: return (1, 0)
    if geno == 3: return (1, 1)

@njit
def parse_segregation(seg):
    if seg == 0: return (0, 0)
    if seg == 1: return (0, 1)
    if seg == 2: return (1, 0)
    if seg == 3: return (1, 1)


@njit
def norm_1D(vect):
    count = 0
    for i in range(len(vect)):
        count += vect[i]
    for i in range(len(vect)):
        vect[i] /= count
@njit
def norm_1D_return(vect):
    count = 0
    for i in range(len(vect)):
        count += vect[i]
    for i in range(len(vect)):
        vect[i] /= count
    return vect

def combineAndConvertToDosage(genoProb1, genoProb2) :
    tmp, nLoci = genoProb1.shape
    dosages = np.full(nLoci, 0, dtype = np.float32)

    for i in range(nLoci):
        vect1 = norm_1D_return(genoProb1[:, i])
        vect2 = norm_1D_return(genoProb2[:, i])

        combined = norm_1D_return(vect1 * vect2) # Not the fastest, but shrug.
        dosages[i] = combined[1] + combined[2] + 2*combined[3]
    return dosages





@njit
def max_multisample(input_mat):
    flattened = input_mat.ravel()

    weights = exp_1D_norm(flattened)
    sample =  weighted_sample_1D(weights)

    max_val = np.max(flattened)
    indexes = np.where(np.abs(flattened - max_val) < 1e-6)[0]
    if(len(indexes) == 0) : print(flattened, max_val, indexes)
    if len(indexes) == 1:
        max_index = indexes[0]
    else:
        max_index = indexes[np.random.randint(0, len(indexes))]
    # if len(flattened) > 4:
    #     print(sample, max_index)
    return sample

@jit(nopython=True, nogil=True) 
def weighted_sample_1D(mat):
    # Get sum of values    
    total = 0
    for i in range(mat.shape[0]):
        total += mat[i]
    value = random.random()*total

    # Select value
    for i in range(mat.shape[0]):
        value -= mat[i]
        if value < 0:
            return i

    return -1


