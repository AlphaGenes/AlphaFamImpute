from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import BasicHMM
from ..tinyhouse import ProbMath
import math

import numpy as np

from numba import jit

def imputeFamFromParentAverage(fam, pedigree) : 

    # Pipeline:
    # 0) We need a segregation tensor.
    # 1) Take all of the children, find the projection of the children up to the parents.
    # 2) Take the parents, and find the joint genotype probabilities.

    # 3) Combine the joint genotype probabilities of the children + parents, and impute back down to the children.

    # STEP 0) We need a segregation tensor.

    segregation = ProbMath.generateSegregation(partial = True) #Using the partial matrix since we aren't using segregation estimates.

    
    # STEP 1) Take all of the children, find the projection of the children up to the parents.

    childGenoProbs = []
    for child in fam.offspring:
        childGenoProbs.append(ProbMath.getGenotypeProbabilities_ind(child))

    projectedGenotypes = []
    allProjected = None
    for i, child in enumerate(fam.offspring):
        projectedGenotypes.append(logPeelUp(childGenoProbs[i], segregation))
        if allProjected is None:
            allProjected = projectedGenotypes[i].copy()
        else:
            allProjected += projectedGenotypes[i]

    
    # STEP 2) Take the parents, and find the joint genotype probabilities.
    
    sireGenoProbs = ProbMath.getGenotypeProbabilities_ind(fam.sire)
    damGenoProbs = ProbMath.getGenotypeProbabilities_ind(fam.dam)
    jointParents = np.log(np.einsum("ai, bi -> abi", sireGenoProbs, damGenoProbs))


    # 3) Combine the joint genotype probabilities of the children + parents, and impute back down to the children.
    for i, child in enumerate(fam.offspring):
        jointParents_combined = jointParents + allProjected - projectedGenotypes[i]
        jointParents_combined = exp_norm(jointParents_combined)
        jointParents_combined /= np.sum(jointParents_combined, (0,1))

        projectedDown = np.einsum("abi, abc -> ci", jointParents_combined, segregation)
        child_combined = projectedDown * childGenoProbs[i]
        child_combined /= np.sum(child_combined, 0)

        child.dosages = np.dot(np.array([0,1,1,2]), child_combined)


@jit(nopython=True)
def exp_norm(mat):
    # Matrix is 4x4xnLoci
    nLoci = mat.shape[2]
    for i in range(nLoci):
        maxVal = 1 #Log of anything between 0-1 will be less than 0. Using 1 as a default.
        for a in range(4):
            for b in range(4):
                if mat[a, b, i] > maxVal or maxVal == 1:
                    maxVal = mat[a, b, i]
        for a in range(4):
            for b in range(4):
                mat[a, b, i] -= maxVal
    return np.exp(mat)


def logPeelUp(genoprobs, segregation):
    jointGenotypes = np.einsum("ci, abc -> abi", genoprobs, segregation)
    return np.log(jointGenotypes)

