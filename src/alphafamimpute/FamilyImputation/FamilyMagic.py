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


def founderImputation(sire, dam, children, preimpute = False):
    
    nChildren = len(children)

    ref_reads = np.full((nChild, nLoci), 0, dtype = np.float32)
    alt_reads = np.full((nChild, nLoci), 0, dtype = np.float32)

    for i in range(nChildren)
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

    perform_updates(child_point_estimates, parent_point_estimates)

def perform_updates(child_point_estimates, parent_point_estimates):
    nChildren = child_point_estimates.shape[0]
    nLoci = child_point_estimates.shape[4]

    # gamma step.
    gamma = np.full((4, 4, nLoci), 0, dtype = np.float32) # Haplotypes. One dimension for each parent. This is not going to scale, but maybe some independence statements?

    gamma[:,:,0] = parents
    for child in range(nChildren):
        gamma[:,:,0] += log_sum_seg(child_point_estimates[child,:, :,:,0])

    for child in range(nChildren):
        aNorm = aTilde[child, :, :, :, 0]/log_sum_seg(aTilde[child, :, :, :, 0])


    for i in range(1, nLoci):
        for child in range(nChildren):
            aTilde[child, :,:,i] = child_point_estimates[child,:, :, :, i] + transmit(log_sum_geno(aNorm[child,:,:,:,i-1] + gamma[:, :, i-1]))

        gamma[:,:,i] = parents
        for child in range(nChildren):
            gamma[:,:,i] += log_sum_seg(aTilde[child, :, :, :, 0])


        for child in range(nChildren):
            aNorm = aTilde[child, :, :, :, i]/log_sum_seg(aTilde[child, :, :, :, i])


@njit
def fill_child_point_estimate(point_estimate, ref_reads, alt_reads) :

    for seg in range(4):
        for sire in range(4):
            for dam in range(4):
                genotype = get_genotype(seg, sire, dam)
                point_estimate[sire, dam,:] = get_genotype_probs(genotype, ref_reads, alt_reads)


@njit
def fill_parent_point_estimate(point_estimate, sire_ref, sire_alt, dam_ref, dam_alt) :
    for sire in range(4):
        sire_0, sire_1 = parse_genotype(parent_genotype)
        sire_geno = sire_0 + sire_1

        for dam in range(4):
            dam_0, dam_1 = parse_genotype(parent_genotype)
            dam_geno = dam_0 + dam_1

            point_estimate[sire, dam,:] = get_genotype_probs(sire_geno, sire_ref, sire_alt)
            point_estimate[sire, dam,:] += get_genotype_probs(dam_geno, dam_ref, dam_alt)


@njit
def fill_child_point_estimate(point_estimate, ref_reads, alt_reads) :

    for parent_genotype in range(4):
        hap_0, hap_1 = parse_genotype(parent_genotype)
        point_estimate[seg, sire, dam,:] = get_genotype_probs(hap_0+hap_1, ref_reads, alt_reads)


@njit
def get_genotype_probs(genotype, ref_reads, alt_reads):
    loge = np.log(0.001)
    log1e = np.log(1-0.001)
    log2 = np.log(.5)

    if genotype == 0: return ref_reads*log1e + alt_reads*loge
    if genotype == 1: return ref_reads*log2 + alt_reads*log2
    if genotype == 2: return ref_reads*log1e + alt_reads*loge


@njit
def get_genotype(seg, sire, dam):

    seg_sire, seg_dam = parse_segregation(seg)

    geno_sire = parse_genotype(sire)[seg_sire]
    geno_dam = parse_genotype(sire)[seg_dam]

    return geno_sire, geno_dam, geno_sire + geno_dam

@njit
def parse_genotype(geno):
    if geno == 0: return (0, 0)
    if geno == 1: return (0, 1)
    if geno == 2: return (1, 0)
    if geno == 3: return (1, 1)

@njit
def parse_segregation(seg):
    if geno == 0: return (0, 0)
    if geno == 1: return (0, 1)
    if geno == 2: return (1, 0)
    if geno == 3: return (1, 1)


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

