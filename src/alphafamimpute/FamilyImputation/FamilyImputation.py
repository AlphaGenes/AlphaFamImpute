from ..tinyhouse import ProbMath
from ..tinyhouse import InputOutput

from . import FamilySingleLocusPeeling
from . import DiploidHMM

from ..tinyhouse import Utils

import math
import random
import numpy as np

from numba import jit, njit
import math


def impute_family(fam, pedigree, rec_rate = None, args = None) :

    # Algorithm outline:
    # 1) Generate a list of high-density (hd) and low-density (ld) offspring. For GBS use all of the offspring.
    # 2) Use the hd offspring to phase the parents. 
    # 3) Impute all of the offspring based on the phased parents.


    # STEP 1: Generate a list of high-density (hd) and low-density (ld) offspring. For GBS use all of the offspring.
    if args.hd_threshold > 0 and not args.gbs:
        hd_children = [off for off in fam.offspring if off.getPercentMissing() <= args.hd_threshold]
        if len(hd_children) == 0:
            print(f"WARNING: There are no high-density children of parents: {fam.sire.idn} and {fam.dam.idn}. Consider using a different high-density threshold. Running with hd_threshold = 0")
            hd_children = fam.offspring
    else:
        hd_children = fam.offspring

    # STEP 2: Phase (and impute) the parents based on the high-density children. 
    parent_geno_probs = parent_phasing(fam.sire, fam.dam, hd_children, rec_rate, args = None)
    
    ProbMath.call_genotype_probs(fam.sire, np.sum(parent_geno_probs, 0), calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)
    ProbMath.call_genotype_probs(fam.dam, np.sum(parent_geno_probs, 1), calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)


    # Step 3: Impute all of the offspring based on the phased parents.

    paternal_probs, maternal_probs = extract_haplotype_probs(parent_geno_probs)
    for child in fam.offspring:
        geno_probs = DiploidHMM.diploidHMM(child, paternal_probs, maternal_probs, args.error, rec_rate, .95)
        ProbMath.call_genotype_probs(child, geno_probs, calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)

# @Utils.time_func("Founder_imputation")
def parent_phasing(sire, dam, children, rec_rate = None, args = None):
    nChildren = len(children)
    nLoci = len(sire.genotypes)

    if rec_rate is None:
        rec_rate = 1.0/nLoci

    # Outline: 
    # 1) We need to setup some data for the actual phasing rounds. 
    # 2) we run the phasing rounds and pass back the results.
    
    # STEP 1: We need to setup some data for the actual phasing rounds. 
    
    # Children
    
    # child_point_estimates: dim_1 = segregation, dim_2 = sire_genotypes, dim_3 = dam_genotypes. 
    child_point_estimates = np.full((nChildren, 4, 4, 4, nLoci), 0, dtype = np.float32) 
    for i, child in enumerate(children):
        child_geno_probs = ProbMath.getGenotypeProbabilities_ind(child, args = args, log=True)
        get_child_point_estimate(child_point_estimates[i,:,:,:,:], child_geno_probs)


    #Parents

    # parent_point_estimates: dim_0 = sire_genotypes, dim_1 = dam_genotypes. 
    parent_point_estimates = np.full((4, 4, nLoci), 0, dtype = np.float32) 
    sire_geno_probs = ProbMath.getGenotypeProbabilities_ind(sire, args = args, log=True)
    dam_geno_probs = ProbMath.getGenotypeProbabilities_ind(dam, args = args, log=True)
    fill_parent_point_estimate(parent_point_estimates, sire_geno_probs, dam_geno_probs)

    # STEP 2: we run the phasing rounds and pass back the results.

    parent_geno_probs = run_phasing(child_point_estimates, parent_point_estimates, rec_rate)
    return parent_geno_probs

@njit
def run_phasing(child_point_estimates, parent_point_estimates, rec_rate):
    nChildren = child_point_estimates.shape[0]
    nLoci = child_point_estimates.shape[4]

    # We do this in a two pass approach -- in the first pass we try and phase and cluster individuals across the chromosome.
    # In the second pass we use the final segregation value and then run a backward pass using that final value.

    # BACKWARD PASS -- Unknown start with phasing. 
    initial_seg = np.full((nChildren, 4), .25, dtype = np.float32)
    child_seg, parent_geno_probs= phasing_round(parent_point_estimates, child_point_estimates, initial_seg, False, rec_rate = rec_rate)

    # FORWARD PASS -- Use called seg at last loci to start phasing children.  
    for i in range(nChildren):
        initial_seg[i, :] = get_transmitted_seg_matrix(child_seg[i, 0], rec_rate)

    child_seg, parent_geno_probs = phasing_round(parent_point_estimates, child_point_estimates, initial_seg, True, rec_rate = rec_rate)
    return parent_geno_probs

@njit
def phasing_round(parent_estimate, child_estimate, initial_seg, forward, rec_rate):
    nChildren = child_estimate.shape[0]
    nLoci = child_estimate.shape[-1]


    # Genotype probabilities for the parents, and called genotypes.
    parent_geno_probs = np.full((4, 4, nLoci), 9, dtype = np.float32)
    # parent_genotypes = np.full((2, nLoci), 9, dtype = np.int64) 


    # Called child segregation, and segregation probabilities.
    child_seg = np.full((nChildren, nLoci), 9, dtype = np.int64)
    backward_seg = np.full((nChildren, 4, nLoci), .25, dtype = np.float32)


    # Setup iteration across the chromosome.
    if forward == True:
        start = 0
        stop = nLoci
        step = 1
    else:
        start = nLoci - 1
        stop = -1 # We need to include 0, so using -1 as the stopping point.
        step = -1 



    first_loci = True
    sire_score = np.full((4, 4), 0, dtype = np.float32)

    projected_seg = np.full((nChildren, 4), .25, dtype = np.float32)
    
    # Temporary values for select_value
    
    select_value_tmp_1d = np.full(4, 0, dtype = np.int64)
    select_value_tmp_2d = np.full((4, 4), 0, dtype = np.int64)
    for i in range(start, stop, step):

        # Transmit the segregation at the previous locus to the current locus. The previous locus is "i-step". Use initial_seg for the first locus.
        for child in range(nChildren):
            if first_loci:
                projected_seg[child,:] = initial_seg[child, :]
            else:
                transmit(backward_seg[child, :, i-step], output = projected_seg[child,:], rec_rate = rec_rate)
        if first_loci:
            first_loci = False

        # Peel up the children to their parents, then select the parent's genotypes.
        sire_score[:,:] = parent_estimate[:, :, i]
        for child in range(nChildren):
            project_child_to_parent(child_estimate[child, :, :, :, i], projected_seg[child,:], sire_score)

        exp_2D_norm(sire_score, parent_geno_probs[:,:,i])
        sire_geno, dam_geno = select_value_2D(sire_score, select_value_tmp_2d)


        # Update the children's segregation values based on the called parental genotypes.
        for child in range(nChildren):
            backward_seg[child, :, i] = np.exp(child_estimate[child, :, sire_geno, dam_geno, i]) * projected_seg[child,:]
            norm_1D(backward_seg[child, :, i]) # This gets passed to the next loci.
            child_seg[child, i] = select_value_1D(backward_seg[child, :, i], select_value_tmp_1d) # This gets saved and returned.

    return child_seg, parent_geno_probs


@njit
def extract_haplotype_probs(joint_geno_probs):
    maternal_probs = np.sum(joint_geno_probs, 0)
    paternal_probs = np.sum(joint_geno_probs, 1)

    return convert_to_dosages(paternal_probs), convert_to_dosages(maternal_probs)


@njit
def convert_to_dosages(geno_probs) :
    nLoci = geno_probs.shape[-1]
    output = np.full((2, nLoci), 0, dtype = np.float32)
    output[0, :] = geno_probs[2,:] + geno_probs[3,:]
    output[1, :] = geno_probs[1,:] + geno_probs[3,:]
    return output


@njit
def get_transmitted_seg_matrix(seg, rec_rate):
    # Take a known segregation value and transmit it to the next loci.
    prev = np.full(4, 0, dtype = np.float32)
    output = np.full(4, 0, dtype = np.float32)
    prev[seg] = 1
    return transmit(prev, output, rec_rate)


@njit
def transmit(vect, output, rec_rate):
    # Transmit a segregation value to the next locus.

    norm_1D(vect)
    e = rec_rate
    e2 = e**2
    e1e = e*(1-e)
    e2i = (1.0-e)**2

    output[0] = e2*vect[3] + e1e*(vect[1] + vect[2]) + e2i*vect[0] 
    output[1] = e2*vect[2] + e1e*(vect[0] + vect[3]) + e2i*vect[1] 
    output[2] = e2*vect[1] + e1e*(vect[0] + vect[3]) + e2i*vect[2] 
    output[3] = e2*vect[0] + e1e*(vect[1] + vect[2]) + e2i*vect[3] 

    return output


@njit
def project_child_to_parent(child_point_estimate, segregation, output) :
    # Estimates the parent's genotypes based on the offspring genotypes.
    for pat in range(4):
        for mat in range(4):
            output[pat, mat] += log_sum_exp_1D(child_point_estimate[:, pat, mat], weights = segregation) # Residual on parent genotypes.
    return output


@njit
def get_child_point_estimate(point_estimate, child_geno_probs) :
    for seg in range(4):
        for sire in range(4):
            for dam in range(4):
                genotype = get_genotype(seg, sire, dam)
                point_estimate[seg, sire, dam,:] = child_geno_probs[genotype, :]


@njit
def fill_parent_point_estimate(point_estimate, sire_geno_probs, dam_geno_probs) :
    for sire in range(4):
        for dam in range(4):
            point_estimate[sire, dam,:] = sire_geno_probs[sire]
            point_estimate[sire, dam,:] += dam_geno_probs[dam]


@njit
def get_genotype(seg, sire, dam):
    # Returns the genotype of an individual based on the parent genotypes and segregation values.
    seg_sire, seg_dam = parse_segregation(seg)

    geno_sire = parse_genotype(sire)[seg_sire]
    geno_dam = parse_genotype(dam)[seg_dam]

    if geno_sire == 0 and geno_dam == 0: return 0
    if geno_sire == 0 and geno_dam == 1: return 1
    if geno_sire == 1 and geno_dam == 0: return 2
    if geno_sire == 1 and geno_dam == 1: return 3
    return 0


@njit
def parse_genotype(geno):
    # Returns the two alleles of an individual based on a genotype index.
    # Genotypes: aa, aA, Aa, AA
    if geno == 0: return (0, 0)
    if geno == 1: return (0, 1)
    if geno == 2: return (1, 0)
    if geno == 3: return (1, 1)


@njit
def parse_segregation(seg):
    # Returns the segregation of an individual based on an index.
    # Segregation: pp, pm, mp, mm

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
def norm_2D(vect):
    count = 0
    for i in range(vect.shape[0]):
        for j in range(vect.shape[1]):
            count += vect[i, j]
    for i in range(vect.shape[0]):
        for j in range(vect.shape[1]):
            vect[i, j] /= count
    

@njit
def select_value_1D(input_mat, tmp):
    # Returns the maximum value of a matrix.
    # If there are multiple possible values, randomly selects one of them.
    max_val = np.max(input_mat)

    n_hits = 0
    for i in range(len(input_mat)):
        if np.abs(input_mat[i] - max_val) < 1e-6:
            tmp[n_hits] = i
            n_hits += 1
    
    if n_hits == 1:
        return tmp[0]
    else:
        return tmp[np.random.randint(n_hits)]

@njit
def select_value_2D(input_mat, tmp):
    # Returns the maximum value of a matrix.
    # If there are multiple possible values, randomly selects one of them.

    max_val = np.max(input_mat)

    n_hits = 0
    for i in range(input_mat.shape[0]):
        for j in range(input_mat.shape[1]):
            if np.abs(input_mat[i, j] - max_val) < 1e-6:
                tmp[n_hits, 0] = i
                tmp[n_hits, 1] = j
                n_hits += 1
        
    if n_hits == 1:
        return tmp[0, 0], tmp[0, 1]
    else:
        index = np.random.randint(n_hits)
        return tmp[index, 0], tmp[index, 1]


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


@jit(nopython=True, nogil = True)
def exp_1D_norm(mat, output):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(mat.shape[0]):
        if mat[a] > maxVal or maxVal == 1:
            maxVal = mat[a]
    
    # Should flag for better numba-ness.
    # output = np.full(mat.shape[0], 0, dtype = np.float32)
    for a in range(mat.shape[0]):
        output[a] = np.exp(mat[a] - maxVal)

    norm_1D(output)
    return output


@jit(nopython=True, nogil = True)
def exp_2D_norm(mat, output):
    # Matrix is 4: Output is to take the exponential of the matrix and normalize each locus. We need to make sure that there are not any overflow values.
    # Note, this changes the matrix in place by a constant.
    maxVal = 1 # Log of anything between 0-1 will be less than 0. Using 1 as a default.
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            if mat[a, b] > maxVal or maxVal == 1:
                maxVal = mat[a, b]
        
    # Should flag for better numba-ness.
    # output = np.full(mat.shape[0], 0, dtype = np.float32)
    for a in range(mat.shape[0]):
        for b in range(mat.shape[1]):
            output[a, b] = np.exp(mat[a, b] - maxVal)

    norm_2D(output)
    return output

