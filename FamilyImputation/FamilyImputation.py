from tinyhouse import HaplotypeOperations
from tinyhouse import BasicHMM
import math


def imputeFam(fam, pedigree) :

    #Pipeline:
    # 0) Get LD and HD children.
    # 1) Take all the children and phase via parents homozygous loci.
    # 2) Take all of the HD children and phase/(impute?) the parents.
    # 3) Take all of the LD children and impute from the parents.

    # STEP 0: Get LD and HD children.

    ldChildren = [off for off in fam.offspring if off.getPercentMissing() > .1]
    hdChildren = [off for off in fam.offspring if off.getPercentMissing() <= .1]

    # STEP 1: Take all the children and phase via parents homozygous loci.

    for ind in hdChildren:
        HaplotypeOperations.setup_individual(ind)
        HaplotypeOperations.fillFromParents(ind)
        HaplotypeOperations.align_individual(ind)

    #STEP 2: Take all of the HD children and phase/(impute?) the parents.

    phaseParentsViaSegregationGroups(fam.sire, fam.dam, hdChildren)

    #STEP 3: Take all of the LD children and impute from the parents.
    for child in ldChildren:
        BasicHMM.diploidHMM(child, fam.sire.haplotypes, fam.dam.haplotypes, 0.01, 1.0/nLoci)


def phaseParentsViaSegregationGroups(sire, dam, children):
    nLoci = length(sire.genotypes)


    # Start at some loci (let's say at the middle of the chromosome)
    # Split the children into the four groups (0,1) X (0,1).
    # Progress across the chromosome and figure out how the groups change over time.

    # STEP 1: Start at some loci (let's say at the middle of the chromosome)
    
    sireStart = getHetLoci(target = sire, alt = dam)
    damStart = getHetLoci(target = dam, alt = sire)
    midpoint = math.floor(nLoci/2)

    # STEP 2: Start at some loci (let's say at the middle of the chromosome)

    groups = [ [[[], []], [[],[]]] for i in range(nLoci)]
    for child in children:
        paternal = child.haplotypes[0][sireStart]
        maternal = child.haplotypes[0][damStart]
        groups[midpoint][paternal][maternal].append(child) #I'm just going to append the child... for reasons.
    # STEP 3: Progress across the chromosome and figure out how the groups change over time.

    for i in range(midpoint + 1, nLoci):
        updateGroups(groups[i-1], groups[i], i-1, i, sire, dam, children)


def updateGroups(currentGroups, nextGroups, currentLoci, nextLoci, sire, dam, children) :

    # Check possible phasings, and correct genotyping errors.
    sireOptions = getOptions(sire.genotypes[nextLoci])
    damOptions = getOptions(dam.genotypes[nextLoci])

    pairOptions = [(op1, op2) for op1 in sireOptions for op2 in damOptions]


    scores = np.full(len(pairOptions), 0, dtype = np.float32)
    for i, pair in enumerate(pairOptions):
        # I don't think we ever need to use the parental phasing here.
        # If the parent is homozygous, we never consider the alternative case where they might be heterozygous/homozygous for the other allele.
        scores[i] = evaluatePair(pair, currentGroups, nextLoci)

    # Pick the best option, and assign the parental haplotypes to that option.
    chosenPair = pairOptions[np.argmin(scores)]
    sirePhase, damPhase = chosenPair
    sire.haplotypes[0][nextLoci] = sirePhase[0]
    sire.haplotypes[1][nextLoci] = sirePhase[1]

    dam.haplotypes[0][nextLoci] = damPhase[0]
    dam.haplotypes[1][nextLoci] = damPhase[1]

    # Re-align the groups around that option.
    
    createNextGroups(chosenPair, currentGroups, nextGroups, nextLoci)

def createNextGroups(pair, currentGroups, nextGroups, loci):
    genotypes = getGenotypesFromPair(pair)

    for i in range(2):
        for j in range(2):
            for ind in currentGroups[i][j]:
                geno = ind.genotypes[loci]
                if geno == genotypes[i][j]:                    
                    nextGroups[i][j].append(ind)

                elif geno == genotypes[1-i][j]:                    
                    nextGroups[1-i][j].append(ind)
                
                elif geno == genotypes[i][1-j]:                    
                    nextGroups[i][1-j].append(ind)
                
                elif geno == genotypes[1-i][1-j]:                    
                    nextGroups[1-i][1-j].append(ind)

                else: # To account for genotyping error.
                    nextGroups[i][j].append(ind)

def getGenotypesFromPair(pair):
    sire, dam = pair
    genotypes = [[9,9],[9,9]]
    for i in range(2):
        for j in range(2):
            genotypes[i][j] = sire[i] + dam[j]
    return genotypes

def evaluatePair(pair, currentGroups, loci):
    # Pair looks like (0, 0), (1, 0)
    
    genotypes = getGenotypesFromPair(pair)

    numErrors = 0
    for i in range(2):
        for j in range(2):
            for ind in currentGroups[i][j]:
                if ind.genotypes[loci] != genotypes[i][j]:
                    numErrors += 1

    return numErrors

def getHetLoci(target, alt):
    nLoci = length(target.genotypes)
    midpoint = math.floor(nLoci/2)
    locus = -1
    i = 0
    while locus < 0 and i < (nLoci/2 + 1):
        forward = min(nLoci - 1, midpoint + i) 
        backward = max(0, midpoint - i) 
        if target.genotypes[forward] == 1 and alt.genotype[forward] != 1:
            locus = forwards
        if target.genotypes[backward] == 1 and alt.genotype[backward] != 1:
            locus = backward
        i += 1

    return locus