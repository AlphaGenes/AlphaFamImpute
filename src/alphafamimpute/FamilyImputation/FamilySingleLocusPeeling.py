from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import ProbMath
import math

import numpy as np

from numba import jit

def impute_from_parent_average(fam, pedigree, args) : 

    # Pipeline:
    # 0) We need a segregation tensor and child/parent genotype probabilities.
    # 1) Project the children up to the sire/dam.
    # 2) Peel the sire/dam down to each of the offspring and call the offspring genotypes
    # 3) Call the parents based on the parents + offspring. 

    # STEP 0) We need a segregation tensor and child/parent genotype probabilities.

    segregation = ProbMath.generateSegregation(partial = True) #Using the partial matrix since we aren't using segregation estimates.
    
    child_probs = []
    for child in fam.offspring:
        child_probs.append(ProbMath.getGenotypeProbabilities_ind(child))

    sire_probs = ProbMath.getGenotypeProbabilities_ind(fam.sire, log = True)
    dam_probs = ProbMath.getGenotypeProbabilities_ind(fam.dam, log = True)

    nLoci = sire_probs.shape[-1]
    
    # STEP 1) Project the children up to the sire/dam.

    child_projection = []
    combined_projection = np.full((4, 4, nLoci), 0, dtype = np.float32)
    for i, child in enumerate(fam.offspring):
        child_projection.append(logPeelUp(child_probs[i], segregation))
        combined_projection += child_projection[i]

    
    # STEP 2) Take the parents, and find the joint genotype probabilities.

    joint_parent_probs = np.full((4, 4, nLoci), 0, dtype = np.float32)
    joint_parent_probs += sire_probs[:, None, :] # Add the sire genotypes.
    joint_parent_probs += dam_probs[None, :, :] # Add the dam genotypes.
    joint_parent_probs += combined_projection # Add the projection from all the offspring.

    for i, child in enumerate(fam.offspring):
        joint_without_child = joint_parent_probs - child_projection[i] # Remove an individual's contribution to their parent's genotypes.
        joint_without_child = exp_with_rescalling(joint_without_child)
        joint_without_child /= np.sum(joint_without_child, (0,1))

        parent_projection = np.einsum("abi, abc -> ci", joint_without_child, segregation)
        child_combined = parent_projection * child_probs[i]
        child_combined /= np.sum(child_combined, 0)

        ProbMath.call_genotype_probs(child, child_combined, calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)

    # STEP 3) Call the parents based on parent data + offspring data.

    parent_probs = exp_with_rescalling(joint_parent_probs)
    parent_probs /= np.sum(joint_parent_probs, (0,1))

    new_sireGenoProbs = np.einsum("abi -> ai", parent_probs)
    new_damGenoProbs = np.einsum("abi -> bi", parent_probs)

    ProbMath.call_genotype_probs(fam.sire, new_sireGenoProbs, calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)
    ProbMath.call_genotype_probs(fam.dam, new_damGenoProbs, calling_threshold = args.calling_threshold, set_genotypes = True, set_dosages = True, set_haplotypes = True)

@jit(nopython=True)
def exp_with_rescalling(mat):
    # Find the exponential of a matrix (up to a multiplicative constant). 
    # This takes into account the fact that some of the values may be very small causing overflow/underflow issues.
    
    nLoci = mat.shape[2]
    for i in range(nLoci):
        first_val = True
        max_val = 1 
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                if mat[a, b, i] > max_val or first_val:
                    max_val = mat[a, b, i]
                if first_val:
                    first_val = False

        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                mat[a, b, i] -= max_val

    return np.exp(mat)


def logPeelUp(probs, segregation):
    # Peels up the genotypes of an individual onto an estimate of their parent's genotypes.
    jointGenotypes = np.einsum("ci, abc -> abi", probs, segregation)
    return np.log(jointGenotypes)

