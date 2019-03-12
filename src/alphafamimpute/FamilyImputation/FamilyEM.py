from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import BasicHMM
from ..tinyhouse import ProbMath
import math

import numpy as np

from numba import jit, njit
import math

def imputeFamUsingFullSibs(fam, pedigree) :

    #Pipeline:
    # 0) Get LD and HD children.
    # 1) Take all the children and phase via parents homozygous loci.
    # 2) Take all of the LD children and impute from the parents.

    # STEP 0: Get LD and HD children.
    nLoci = len(fam.sire.genotypes)

    ldChildren = [off for off in fam.offspring if off.getPercentMissing() > .1]
    hdChildren = [off for off in fam.offspring if off.getPercentMissing() <= .1]

    childDosages = np.full((len(ldChildren), nLoci), 0, dtype = np.float32)

    for cycle in range(10) :
         runImputationRound(fam, ldChildren, hdChildren)
         for i, child in enumerate(ldChildren):
            childDosages[i,:] += child.dosages

    childDosages /= 10
    for i, child in enumerate(ldChildren):
        child.dosages = childDosages[i,:]

def runImputationRound(fam, ldChildren, hdChildren) :
    #STEP 1: Take all of the HD children and phase/(impute?) the parents.

    sireHaplotypes, damHaplotypes = phaseParentsViaEM(fam.sire, fam.dam, hdChildren)

    # Just re-align the parents sowe get imputed genotypes from them on the output.
    # for ind in [fam.sire, fam.dam]:
    #     HaplotypeOperations.align_individual(ind)


    #STEP 2: Take all of the LD children and impute from the parents.
    nLoci = len(fam.sire.genotypes)
    

    for child in ldChildren:
        # imputeIndividual(child, np.round(sireHaplotypes), np.round(damHaplotypes))
        BasicHMM.diploidHMM(child, np.round(sireHaplotypes), np.round(damHaplotypes), 0.01, 1.0/nLoci)


def phaseParentsViaEM(sire, dam, children):
    # Pipeline:
    # 0) Initialize founder haplotypes. 

    sire_genotypeProbabilities = ProbMath.getGenotypeProbabilities_ind(sire)
    dam_genotypeProbabilities = ProbMath.getGenotypeProbabilities_ind(dam)

    nLoci = sire_genotypeProbabilities.shape[1]

    sireHaplotypes = np.full((2, nLoci), 0, dtype = np.float32)
    damHaplotypes = np.full((2, nLoci), 0, dtype = np.float32)

    sireHaplotypes[0,:] = sire_genotypeProbabilities[1,:] + sire_genotypeProbabilities[3,:]
    sireHaplotypes[1,:] = sire_genotypeProbabilities[2,:] + sire_genotypeProbabilities[3,:]

    damHaplotypes[0, :] = dam_genotypeProbabilities[1,:] + dam_genotypeProbabilities[3,:]
    damHaplotypes[1, :] = dam_genotypeProbabilities[2,:] + dam_genotypeProbabilities[3,:]

    # Add in some jitter for the haplotype assignements.
    sireHaplotypes = .7 * sireHaplotypes + 0.15
    sireHaplotypes = sireHaplotypes + (1 - 2*np.random.random(sireHaplotypes.shape)) * 0.1 # Just a little bit of jitter. Could probably initialize better.
    
    damHaplotypes = .7 * damHaplotypes + 0.15
    damHaplotypes = damHaplotypes + (1 - 2*np.random.random(damHaplotypes.shape)) * 0.1 # Just a little bit of jitter. Could probably initialize better.

    nChildren = len(children)
    # print("sire", sireHaplotypes)
    # print("dam", damHaplotypes)

    for i in range(100):
        # 1) Loop to perform haplotype assignments based on current haplotypes.
    
        segregations = np.full((nChildren, 2, 2, nLoci), 0, dtype = np.float32)

        genotypeProbabilities = np.full((nChildren, 4, nLoci), 0, dtype = np.float32)
        for i, child in enumerate(children):
            genotypeProbabilities[i,:,:] = ProbMath.getGenotypeProbabilities_ind(child)

        for i, child in enumerate(children):
            estimateSegregation(segregations[i,:,:], genotypeProbabilities[i,:,:], sireHaplotypes, damHaplotypes)


        # 2) Loop to re-estimate the founder haplotypes based on assignements.
    
        sireHaplotypes, damHaplotypes = estimateFounders(segregations, genotypeProbabilities, sireHaplotypes, damHaplotypes, sire_genotypeProbabilities, dam_genotypeProbabilities)

        # print("sire", sireHaplotypes)
        # print("dam", damHaplotypes)
    #Now we're going to call...

    # sire.haplotypes[0][:] = np.round(sireHaplotypes[0,:])
    # sire.haplotypes[1][:] = np.round(sireHaplotypes[1,:])
    # dam.haplotypes[0][:] = np.round(damHaplotypes[0,:])
    # dam.haplotypes[1][:] = np.round(damHaplotypes[1,:])

    return sireHaplotypes, damHaplotypes

@njit
def estimateSegregation(segregation, genotypeProbabilities, sireHaplotypes, damHaplotypes):
    nLoci = segregation.shape[2]
    pointEstimates = np.full((2, 2, nLoci), 0, dtype = np.float32) 

    for i in range(nLoci):
        for sireHap in range(2):
            for damHap in range(2):
                # Flag: This may be a place where we are playing too fast and loose with a normalization constant.
                p1 = sireHaplotypes[sireHap, i]
                p2 = damHaplotypes[damHap, i]
                hapsToGeno = np.array([(1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2], dtype = np.float32)
                score = np.sum(hapsToGeno * genotypeProbabilities[:,i])
                pointEstimates[sireHap, damHap, i] = score

    recombinationRate = np.full(nLoci, 1.0/nLoci, dtype = np.float32)

    segregation[:,:,:] = BasicHMM.diploidForwardBackward(pointEstimates, recombinationRate = recombinationRate)

@njit
def estimateFounders(segregations, genotypeProbabilities, sireHaplotypes, damHaplotypes, sire_genotypeProbabilities, dam_genotypeProbabilities):
    nChildren, tmp, tmp2, nLoci = segregations.shape

    sireHaplotypes_new = np.full((2, nLoci), 0.1, dtype = np.float32)
    sireHaplotypes_new_counts = np.full((2, nLoci), 0.2, dtype = np.float32)

    damHaplotypes_new = np.full((2, nLoci), 0.1, dtype = np.float32)
    damHaplotypes_new_counts = np.full((2, nLoci), 0.2, dtype = np.float32)

    # Sire update:
    for i in range(nLoci):
        p1 = sireHaplotypes[0,i]
        p2 = sireHaplotypes[1,i]
        values = np.array([(1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2], dtype = np.float32) * sire_genotypeProbabilities[:,i]
        values = values/np.sum(values)
        sire_dosage = values[2] + values[3]
        dam_dosage = values[1] + values[3]
        sireHaplotypes_new[0, i] += sire_dosage
        sireHaplotypes_new_counts[0, i] += 1

        sireHaplotypes_new[1, i] += dam_dosage
        sireHaplotypes_new_counts[1, i] += 1
    
    # Dam updates
    for i in range(nLoci):
        p1 = damHaplotypes[0,i]
        p2 = damHaplotypes[1,i]
        values = np.array([(1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2], dtype = np.float32) * dam_genotypeProbabilities[:,i]
        values = values/np.sum(values)
        sire_dosage = values[2] + values[3]
        dam_dosage = values[1] + values[3]
        damHaplotypes_new[0, i] += sire_dosage
        damHaplotypes_new_counts[0, i] += 1

        damHaplotypes_new[1, i] += dam_dosage
        damHaplotypes_new_counts[1, i] += 1

    for child in range(nChildren):
        for i in range(nLoci):
            for sireHap in range(2):
                for damHap in range(2):
                    # Flag: This may be a place where we are playing too fast and loose with a normalization constant.
                    p1 = sireHaplotypes[sireHap,i]
                    p2 = damHaplotypes[damHap,i]
                    values = np.array([(1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2], dtype = np.float32) * genotypeProbabilities[child,:,i]
                    values = values/np.sum(values)
                    sire_dosage = values[2] + values[3]
                    dam_dosage = values[1] + values[3]

                    sireHaplotypes_new[sireHap, i] += segregations[child,sireHap,damHap,i]*sire_dosage
                    sireHaplotypes_new_counts[sireHap, i] += segregations[child,sireHap,damHap,i]

                    damHaplotypes_new[damHap, i] += segregations[child,sireHap,damHap,i]*dam_dosage
                    damHaplotypes_new_counts[damHap, i] += segregations[child,sireHap,damHap,i]
    # print("counts")
    # print(sireHaplotypes_new)
    # print(sireHaplotypes_new_counts)
    # print(damHaplotypes_new)
    # print(damHaplotypes_new_counts)

    return sireHaplotypes_new/sireHaplotypes_new_counts, damHaplotypes_new/damHaplotypes_new_counts


def imputeIndividual(ind, sireHaplotypes, damHaplotypes):
    nLoci = len(sireHaplotypes[0])
    # Take an individual, get their genotype probabilities. 
    genotypeProbabilities = ProbMath.getGenotypeProbabilities_ind(ind)

    # Use the genotype probabilities to generate segregation estimates.
    segregation = np.full((2, 2, nLoci), 0, dtype = np.float32) 
    estimateSegregation(segregation, genotypeProbabilities, sireHaplotypes, damHaplotypes)

    # Use the segregation estimates to re-estimate the individuals genotypes and turn that into dosages.
    ind.dosages = getDosages(segregation, genotypeProbabilities, sireHaplotypes, damHaplotypes)

@njit
def getDosages(segregation, genotypeProbabilities, sireHaplotypes, damHaplotypes):
    tmp, nLoci = genotypeProbabilities.shape
    dosages = np.full(nLoci, 0, dtype = np.float32)
    for i in range(nLoci):
        for sireHap in range(2):
            for damHap in range(2):
                # Flag: This may be a place where we are playing too fast and loose with a normalization constant.
                p1 = sireHaplotypes[sireHap,i]
                p2 = damHaplotypes[damHap,i]
                values = np.array([(1-p1)*(1-p2), (1-p1)*p2, p1*(1-p2), p1*p2], dtype = np.float32) * genotypeProbabilities[:,i]
                values = values/np.sum(values)
                dosage = values[1] + values[2] + 2*values[3]
                dosages[i] += dosage * segregation[sireHap, damHap, i]
    return dosages








































