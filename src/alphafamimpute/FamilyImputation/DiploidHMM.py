from numba import jit
import numpy as np
from ..tinyhouse import ProbMath


def diploidHMM(ind, paternal_probs, maternal_probs, error, recombinationRate, calling_threshold = .95): 

    nLoci = len(ind.genotypes)

    # !!!! NEED TO MAKE SURE SOURCE HAPLOTYPES ARE ALL NON MISSING!!!
    if type(error) is float:
        error = np.full(nLoci, error, dtype = np.float32)

    if type(recombinationRate) is float:
        recombinationRate = np.full(nLoci, recombinationRate, dtype = np.float32)
    # !!!! May need to have marker specific error/recombination rates.

    paternal_called = np.empty(paternal_probs.shape, dtype = np.float32)
    for i in range(paternal_called.shape[0]):
        paternal_called[i,:] = call_haplotypes(paternal_probs[i,:], calling_threshold)

    maternal_called = np.empty(maternal_probs.shape, dtype = np.float32)
    for i in range(maternal_called.shape[0]):
        maternal_called[i,:] = call_haplotypes(maternal_probs[i,:], calling_threshold)

    non_missing_loci = np.where(np.all(paternal_called != 9, axis = 0) & np.all(maternal_called != 9, axis = 0))[0]
    missing_loci = np.where(~ (np.all(paternal_called != 9, axis = 0) & np.all(maternal_called != 9, axis = 0)))[0]
    ###Construct penetrance values

    probs = ProbMath.getGenotypeProbabilities_ind(ind)

    print(probs[:,0:5])
    pointEst = getDiploidPointEstimates_probs(probs, paternal_called, maternal_called, error, non_missing_loci)

    
    ### Run forward-backward algorithm on penetrance values.
    hapEst = diploidForwardBackward(pointEst, recombinationRate)

    geno_probs = get_diploid_geno_probs(hapEst, paternal_probs, maternal_probs)

    return geno_probs
    # dosages = geno_probs[1,:] + geno_probs[2,:] + 2*geno_probs[3,:]
    # ind.dosages = dosages


def get_multi_locus_estimate(hapEst, joint_parent_probs, child_probs):

    segregation_tensor = ProbMath.generateSegregation(partial = False) 
    nLoci = hapEst.shape[-1]
    segregation = np.full((4, nLoci), 0, dtype = np.float32)
    segregation[0, :] = hapEst[0, 0, :]
    segregation[1, :] = hapEst[0, 1, :]
    segregation[2, :] = hapEst[1, 0, :]
    segregation[3, :] = hapEst[1, 1, :]

    child_anterior = np.einsum("abi, abcd, di-> ci", joint_parent_probs, segregation_tensor, segregation)
    child_combined = child_anterior * child_probs
    child_combined /= np.sum(child_combined, 0)

    return child_combined


@jit(nopython=True)
def call_haplotypes(hap, threshold):
    nLoci = len(hap)
    output = np.full(nLoci, 9, dtype = np.int8)
    for i in range(nLoci):
        if hap[i] > threshold : output[i] = 1
        if hap[i] < 1-threshold : output[i] = 0
    return output


@jit(nopython=True)
def getDiploidDosages(hapEst, paternalHaplotypes, maternalHaplotypes):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape
    dosages = np.full(nLoci, 0, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                dosages[i] += hapEst[j,k,i]*(paternalHaplotypes[j,i] + maternalHaplotypes[k,i])
    return dosages


@jit(nopython=True)
def get_diploid_geno_probs(hapEst, paternalHaplotypes, maternalHaplotypes):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape
    geno_probs = np.full((4, nLoci), 0, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                pat_a = 1 - paternalHaplotypes[j,i]
                pat_A = paternalHaplotypes[j,i]
                mat_a = 1 - maternalHaplotypes[k,i]
                mat_A = maternalHaplotypes[k,i]

                geno_probs[0, i] += hapEst[j,k,i]*(pat_a*mat_a)
                geno_probs[1, i] += hapEst[j,k,i]*(pat_a*mat_A)
                geno_probs[2, i] += hapEst[j,k,i]*(pat_A*mat_a)
                geno_probs[3, i] += hapEst[j,k,i]*(pat_A*mat_A)
    return geno_probs



@jit(nopython=True)
def getDiploidPointEstimates(indGeno, indPatHap, indMatHap, paternalHaplotypes, maternalHaplotypes, error):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    pointEst = np.full((nPat, nMat, nLoci), 1, dtype = np.float32)
    for i in range(nLoci):
        if indGeno[i] != 9:
            for j in range(nPat):
                for k in range(nMat):
                    # Seperate Phased vs non phased loci 
                    if indPatHap[i] != 9 and indMatHap[i] != 9:
                        value = 1
                        if indPatHap[i] == paternalHaplotypes[j, i]:
                            value *= (1-error[i])
                        else:
                            value *= error[i]
                        if indMatHap[i] == maternalHaplotypes[k, i]:
                            value *= (1-error[i])
                        else:
                            value *= error[i]
                        pointEst[j,k,i] = value
                    else:
                        #I think this shouldn't be too horrible.
                        sourceGeno = paternalHaplotypes[j, i] + maternalHaplotypes[k,i]
                        if sourceGeno == indGeno[i]:
                            pointEst[j,k, i] = 1-error[i]*error[i]
                        else:
                            pointEst[j,k,i] = error[i]*error[i]
    return pointEst

@jit(nopython=True)
def getDiploidPointEstimates_probs(indProbs, paternalHaplotypes, maternalHaplotypes, error, loci):
    nPat, nLoci = paternalHaplotypes.shape
    nMat, nLoci = maternalHaplotypes.shape

    pointEst = np.full((nPat, nMat, nLoci), 1, dtype = np.float32)
    for i in loci:
        for j in range(nPat):
            for k in range(nMat):
                # I'm just going to be super explicit here. 
                p_aa = indProbs[0, i]
                p_aA = indProbs[1, i]
                p_Aa = indProbs[2, i]
                p_AA = indProbs[3, i]
                e = error[i]
                if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 0:
                    value = p_aa*(1-e)**2 + (p_aA + p_Aa)*e*(1-e) + p_AA*e**2

                if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 0:
                    value = p_Aa*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_aA*e**2

                if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 1:
                    value = p_aA*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_Aa*e**2

                if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 1:
                    value = p_AA*(1-e)**2  + (p_aA + p_Aa)*e*(1-e) + p_aa*e**2

                pointEst[j,k,i] = value
    return pointEst


@jit(nopython=True)
def diploidForwardBackward(pointEst, recombinationRate) :
    #This is probably way more fancy than it needs to be -- particularly it's low memory impact, but I think it works.
    nPat, nMat, nLoci = pointEst.shape

    est = np.full(pointEst.shape, 1, dtype = np.float32)
    for i in range(nLoci):
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] = pointEst[j,k,i]


    tmp = np.full((nPat, nMat), 0, dtype = np.float32)
    new = np.full((nPat, nMat), 0, dtype = np.float32)
    prev = np.full((nPat, nMat), .25, dtype = np.float32)

    pat = np.full(nPat, 0, dtype = np.float32)
    mat = np.full(nMat, 0, dtype = np.float32)

    for i in range(1, nLoci):
        e = recombinationRate[i]
        e1e = e*(1-e)
        e2 = e*e
        e2m1 = (1-e)**2
        #Although annoying, the loops here are much faster for small number of haplotypes.

        #Get estimate at this loci and normalize.
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = prev[j,k]*pointEst[j,k,i-1]
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += tmp[j, k]
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = tmp[j,k]/sum_j

        #Get haplotype specific recombinations
        pat[:] = 0
        mat[:] = 0
        for j in range(nPat):
            for k in range(nMat):
                pat[j] += tmp[j,k]
                mat[k] += tmp[j,k]

        #Account for recombination rate
        for j in range(nPat):
            for k in range(nMat):
                # new[j, k] = tmp[j, k]*e2m1 + e1e*pat[j] + e1e*mat[k] + e2
                new[j,k] = tmp[j, k]*e2m1 + e1e*pat[j]/nPat + e1e*mat[k]/nMat + e2/(nMat*nPat)
                # valInbred = (1-e)*tmp[j, k]
                # if j == k: valInbred += e/nPat
                # new[j,k] = (1-I)*valOutbred + I*valInbred

        #Add to est
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] *= new[j, k]
        prev = new

    prev = np.full((nPat, nMat), 1, dtype = np.float32)
    for i in range(nLoci-2, -1, -1): #zero indexed then minus one since we skip the boundary.
        #I need better naming comditions.
        e = recombinationRate[i+1]
        e1e = (1-e)*e
        e2 = e*e
        e2m1 = (1-e)**2

        #Although annoying, the loops here are much faster for small number of haplotypes.

        #Get estimate at this loci and normalize.
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = prev[j,k]*pointEst[j,k,i+1]
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += tmp[j, k]
        for j in range(nPat):
            for k in range(nMat):
                tmp[j,k] = tmp[j,k]/sum_j

        #Get haplotype specific recombinations
        pat[:] = 0
        mat[:] = 0
        for j in range(nPat):
            for k in range(nMat):
                pat[j] += tmp[j,k]
                mat[k] += tmp[j,k]

        #Account for recombination rate
        for j in range(nPat):
            for k in range(nMat):
                # new[j, k] = tmp[j, k]*e2m1 + e1e*pat[j] + e1e*mat[k] + e2
                new[j,k] = tmp[j, k]*e2m1 + e1e*pat[j]/nPat + e1e*mat[k]/nMat + e2/(nMat*nPat)
                # valInbred = (1-e)*tmp[j, k]
                # if j == k: valInbred += e/nPat
                # new[j,k] = (1-I)*valOutbred + I*valInbred

        #Add to est
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] *= new[j, k]
        prev = new

    for i in range(nLoci):
        sum_j = 0
        for j in range(nPat):
            for k in range(nMat):
                sum_j += est[j,k,i]
        for j in range(nPat):
            for k in range(nMat):
                est[j,k,i] = est[j,k,i]/sum_j

    return(est)







