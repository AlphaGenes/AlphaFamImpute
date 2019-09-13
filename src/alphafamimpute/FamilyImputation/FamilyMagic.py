from ..tinyhouse import HaplotypeOperations
from ..tinyhouse import BasicHMM
from ..tinyhouse import ProbMath
from ..tinyhouse import InputOutput
from . import FamilySingleLocusPeeling
import math

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
    if not args.gbs :
        ldChildren = [off for off in fam.offspring if off.getPercentMissing() > .1]
        hdChildren = [off for off in fam.offspring if off.getPercentMissing() <= .1]
    if args.gbs:
        ldChildren = fam.offspring
        hdChildren = fam.offspring

    if not args.usegenoprobs:
        # childDosages is here to store the children's dosages to prevent them from being over-written.
        childDosages = np.full((len(ldChildren), nLoci), 0, dtype = np.float32)

        nIterations = args.niter
        for cycle in range(nIterations) :
             runImputationRound(fam, ldChildren, hdChildren, callingMethod = "dosages", preimpute = args.preimpute)
             for i, child in enumerate(ldChildren):
                childDosages[i,:] += child.dosages

        childDosages /= nIterations
        for i, child in enumerate(ldChildren):
            child.dosages = childDosages[i,:]

    if args.usegenoprobs:
        childGenoProbs = np.full((len(ldChildren), 4, nLoci), 0, dtype = np.float32)

        nIterations = args.niter
        for cycle in range(nIterations) :
             runImputationRound(fam, ldChildren, hdChildren, callingMethod = "probabilities", preimpute = args.preimpute)
             for i, child in enumerate(ldChildren):
                childGenoProbs[i,:,:] += child.info # Child genotype probabilities get passed to info.

        childGenoProbs /= nIterations
        for i, child in enumerate(ldChildren):
            child.dosages = combineAndConvertToDosage(childGenoProbs[i,:,:], ProbMath.getGenotypeProbabilities_ind(child))


def runImputationRound(fam, ldChildren, hdChildren, callingMethod = "dosages", preimpute = False) :
    #STEP 1: Take all of the HD children and phase/(impute?) the parents.

    sireHaplotypes, damHaplotypes = phaseParentsViaEM(fam.sire, fam.dam, hdChildren, preimpute = preimpute)

    # Just re-align the parents to get imputed genotypes from them on the output.
    # This got taken out because we run the peeling cycle multiple times. Probably should just do something with the dosages.
    # for ind in [fam.sire, fam.dam]:
    #     HaplotypeOperations.align_individual(ind)


    #STEP 2: Take all of the LD children and impute from the parents.
    nLoci = len(fam.sire.genotypes)
    

    for child in ldChildren:
        # imputeIndividual(child, np.round(sireHaplotypes), np.round(damHaplotypes))
        BasicHMM.diploidHMM(child, np.round(sireHaplotypes), np.round(damHaplotypes), 0.01, 1.0/nLoci, useCalledHaps = False, callingMethod = callingMethod)


def phaseParentsViaEM(sire, dam, children, preimpute = False):
    args = InputOutput.args

    # Pipeline:
    # 0) Initialize founder haplotypes. 

    if preimpute:
        sire_genotypeProbabilities, dam_genotypeProbabilities = FamilySingleLocusPeeling.getParentalGenotypesWithChildren(sire, dam, children)
    else:
        sire_genotypeProbabilities = ProbMath.getGenotypeProbabilities_ind(sire)
        dam_genotypeProbabilities = ProbMath.getGenotypeProbabilities_ind(dam)

    nLoci = sire_genotypeProbabilities.shape[1]

    sireHaplotypes = np.full((2, nLoci), 0, dtype = np.float32)
    damHaplotypes = np.full((2, nLoci), 0, dtype = np.float32)

    jitter = args.jitter

    sireHaplotypes[0,:] = sire_genotypeProbabilities[1,:] + sire_genotypeProbabilities[3,:] # Prob the sire is aA or AA
    sireHaplotypes[1,:] = sire_genotypeProbabilities[2,:] + sire_genotypeProbabilities[3,:] # Prob the sire is Aa or AA

    damHaplotypes[0, :] = dam_genotypeProbabilities[1,:] + dam_genotypeProbabilities[3,:]
    damHaplotypes[1, :] = dam_genotypeProbabilities[2,:] + dam_genotypeProbabilities[3,:]

    # Add in some jitter for the haplotype assignements.
    sireHaplotypes = ((1-jitter*2) * sireHaplotypes + jitter) + (1 - 2*np.random.random(sireHaplotypes.shape)) * 2/3*jitter # (0,1) -> (.15, .85) + random noise -> (.05, .95)
    damHaplotypes =  ((1-jitter*2) * damHaplotypes + jitter) + (1 - 2*np.random.random(damHaplotypes.shape)) * 2/3*jitter # (0,1) -> (.15, .85) + random noise -> (.05, .95)

    nChildren = len(children)

    nCycles = args.ncycles

    # Step 0b) Construct genotype probabilities for the children. Note: These are fixed for all the iterations.
    genotypeProbabilities = np.full((nChildren, 4, nLoci), 0, dtype = np.float32)
    for i, child in enumerate(children):
        genotypeProbabilities[i,:,:] = ProbMath.getGenotypeProbabilities_ind(child)

    # Step 1 + 2: Estimate children based on parents. Estimate parents based on children. 
    for i in range(nCycles):
        # 1) Loop to perform haplotype assignments based on current haplotypes.
    
        segregations = np.full((nChildren, 2, 2, nLoci), 0, dtype = np.float32)
        for i, child in enumerate(children):
            estimateSegregation(segregations[i,:,:], genotypeProbabilities[i,:,:], sireHaplotypes, damHaplotypes)

        # 2) Loop to re-estimate the founder haplotypes based on assignements.
            sireHaplotypes, damHaplotypes = estimateFounders(segregations, genotypeProbabilities, sireHaplotypes, damHaplotypes, sire_genotypeProbabilities, dam_genotypeProbabilities)

    return sireHaplotypes, damHaplotypes

@njit
def estimateSegregation(segregation, genotypeProbabilities, sireHaplotypes, damHaplotypes):
    nLoci = segregation.shape[2]
    pointEstimates = np.full((2, 2, nLoci), 0, dtype = np.float32) 

    # Construct point estimates, by comparing sire + dam haplotypes.
    for i in range(nLoci):
        for sireHap in range(2):
            for damHap in range(2):
                # Flag: This may be a place where we are playing too fast and loose with a normalization constant.
                p1 = sireHaplotypes[sireHap, i]
                p2 = damHaplotypes[damHap, i]

                p_aa = genotypeProbabilities[0, i]
                p_aA = genotypeProbabilities[1, i] 
                p_Aa = genotypeProbabilities[2, i]
                p_AA = genotypeProbabilities[3, i]

                # I am reasonable certain that this is right.
                # p(aa | inheriting sireHap + damHap) = (1-p1)*(1-p2)
                # We are calculating p(aa | inheritance, data) = p(aa|data)*p(aa|inheritance).

                score = p_aa*(1-p1)*(1-p2) + p_aA*(1-p1)*p2 + p_Aa*p1*(1-p2) + p_AA*p1*p2 
                pointEstimates[sireHap, damHap, i] = score

    recombinationRate = np.full(nLoci, 1.0/nLoci, dtype = np.float32)

    # Run HMM on point estimates to get smoothed assignments.
    segregation[:,:,:] = BasicHMM.diploidForwardBackward(pointEstimates, recombinationRate = recombinationRate)

@njit
def estimateFounders(segregations, genotypeProbabilities, sireHaplotypes, damHaplotypes, sire_genotypeProbabilities, dam_genotypeProbabilities):
    nChildren, tmp, tmp2, nLoci = segregations.shape

    # The .1 and .2 are weak priors.
    sireHaplotypes_new = np.full((2, nLoci), 0.1, dtype = np.float32)
    sireHaplotypes_new_counts = np.full((2, nLoci), 0.2, dtype = np.float32)

    damHaplotypes_new = np.full((2, nLoci), 0.1, dtype = np.float32)
    damHaplotypes_new_counts = np.full((2, nLoci), 0.2, dtype = np.float32)

    values = np.full(4, 0, dtype = np.float32)


    # Sire update:
    for i in range(nLoci):
        p1 = sireHaplotypes[0,i]
        p2 = sireHaplotypes[1,i]

        # This is probably a speed bottleneck
        values[0]= sire_genotypeProbabilities[0,i] * (1-p1)*(1-p2)
        values[1]= sire_genotypeProbabilities[1,i] * (1-p1)*p2
        values[2]= sire_genotypeProbabilities[2,i] * p1*(1-p2)
        values[3]= sire_genotypeProbabilities[3,i] * p1*p2
        norm_1D(values)

        sire_dosage = values[2] + values[3] # This is the expected allele value they recieved from their sire.
        dam_dosage = values[1] + values[3] # This is the expected allele value they recieved from their dam.
        sireHaplotypes_new[0, i] += sire_dosage
        sireHaplotypes_new_counts[0, i] += 1

        sireHaplotypes_new[1, i] += dam_dosage
        sireHaplotypes_new_counts[1, i] += 1
    
    # Dam updates
    for i in range(nLoci):
        p1 = damHaplotypes[0,i]
        p2 = damHaplotypes[1,i]
        values[0]= dam_genotypeProbabilities[0,i] * (1-p1)*(1-p2)
        values[1]= dam_genotypeProbabilities[1,i] * (1-p1)*p2
        values[2]= dam_genotypeProbabilities[2,i] * p1*(1-p2)
        values[3]= dam_genotypeProbabilities[3,i] * p1*p2
        norm_1D(values)
        sire_dosage = values[2] + values[3]
        dam_dosage = values[1] + values[3]
        damHaplotypes_new[0, i] += sire_dosage
        damHaplotypes_new_counts[0, i] += 1

        damHaplotypes_new[1, i] += dam_dosage
        damHaplotypes_new_counts[1, i] += 1

    valueArray = np.full((2, 2, nLoci, 4), 0, dtype = np.float32)
    for i in range(nLoci):
        for sireHap in range(2):
            for damHap in range(2):
                p1 = sireHaplotypes[sireHap,i]
                p2 = damHaplotypes[damHap,i]

                # This produces the genotype probabilities for an offspring, conditional on having a given haplotype.
                valueArray[sireHap, damHap, i, 0] = (1-p1)*(1-p2)
                valueArray[sireHap, damHap, i, 1] = (1-p1)*p2
                valueArray[sireHap, damHap, i, 2] = p1*(1-p2)
                valueArray[sireHap, damHap, i, 3] = p1*p2

    for child in range(nChildren):
        for i in range(nLoci):
            for sireHap in range(2):
                for damHap in range(2):
                    # Flag: This may be a place where we are playing too fast and loose with a normalization constant.
                    # p1 = sireHaplotypes[sireHap,i]
                    # p2 = damHaplotypes[damHap,i]
                    for j in range(4):
                        values[j] = valueArray[sireHap, damHap, i, j] * genotypeProbabilities[child,j, i]
                    norm_1D(values)

                    sire_dosage = values[2] + values[3]
                    dam_dosage = values[1] + values[3]

                    sireHaplotypes_new[sireHap, i] += segregations[child,sireHap,damHap,i]*sire_dosage
                    sireHaplotypes_new_counts[sireHap, i] += segregations[child,sireHap,damHap,i]

                    damHaplotypes_new[damHap, i] += segregations[child,sireHap,damHap,i]*dam_dosage
                    damHaplotypes_new_counts[damHap, i] += segregations[child,sireHap,damHap,i]

    return sireHaplotypes_new/sireHaplotypes_new_counts, damHaplotypes_new/damHaplotypes_new_counts

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

