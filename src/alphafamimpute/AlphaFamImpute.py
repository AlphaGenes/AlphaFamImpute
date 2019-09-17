from .tinyhouse import Pedigree
from .tinyhouse import InputOutput
from .FamilyImputation import FamilyImputation
from .FamilyImputation import FamilySingleLocusPeeling
from .FamilyImputation import FamilyEM
from .FamilyImputation import FamilyMagic
import numpy as np
import argparse

try:
    profile
except:
    def profile(x): 
        return x

def getArgs() :
    parser = argparse.ArgumentParser(description='')
    core_parser = parser.add_argument_group("Core arguments")
    core_parser.add_argument('-out', required=True, type=str, help='The output file prefix.')
    InputOutput.addInputFileParser(parser)
    
    famParser = parser.add_argument_group("AlphaFamImpute Arguments")

    famParser.add_argument('-extphase', action='store_true', required=False, help='Include when using an external phase file for the parents. Skips the fullsib phase algorithm and just runs the HMM.')
    famParser.add_argument('-parentaverage', action='store_true', required=False, help='Include to impute based on parent average genotype. This runs single locus peeling on the parents to fill in any missing genotypes, then peels down to each offspring.')
    famParser.add_argument('-em', action='store_true', required=False, help='Include to impute based on parent average genotype. This runs single locus peeling on the parents to fill in any missing genotypes, then peels down to each offspring.')
    famParser.add_argument('-gbs', action='store_true', required=False, help='Flag to do some more fully probibilistic calculations for gbs data.')
    famParser.add_argument('-magic', action='store_true', required=False, help='Flag to run the magic impute algorithm.')
    famParser.add_argument('-usegenoprobs', action='store_true', required=False, help=argparse.SUPPRESS) #help='Flag to use the individual\'s genotype probabilities for calculating dosages.')
    famParser.add_argument('-preimpute', action='store_true', required=False, help=argparse.SUPPRESS), #help='Flag to use the children\'s genotypes to pre-impute the parent\'s genotypes (via single locus peeling).')
    
    famParser.add_argument('-ncycles', default=30, required=False, type=int, help='Number of EM cycles. Only has an effect with EM mode, Default = 30.')
    famParser.add_argument('-niter', default=10, required=False, type=int, help='Number of full iterations. Only has an effect with EM mode, Default = 10.')
    famParser.add_argument('-jitter', default=0.01, required=False, type=float, help='Jitter amount for the initial haplotypes. Only has an effect with EM mode. Default = 0.01.')

    return InputOutput.parseArgs("AlphaFamImpute", parser)


@profile
def main():
    args = getArgs() 
    pedigree = Pedigree.Pedigree() 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True, reads = True)

    for fam in pedigree.getFamilies() :
        if args.magic:
            FamilyMagic.imputeFamUsingFullSibs(fam, pedigree, args)
        elif args.extphase :
            FamilyImputation.imputeFamFromPhasedParents(fam, pedigree)
        elif args.parentaverage:
            FamilySingleLocusPeeling.imputeFamFromParentAverage(fam, pedigree)
        elif args.em:
            FamilyEM.imputeFamUsingFullSibs(fam, pedigree, args)
        else:
            FamilyImputation.imputeFamUsingFullSibs(fam, pedigree)

    pedigree.writePhase(args.out + ".phase")
    pedigree.writeGenotypes(args.out + ".genotypes")
    pedigree.writeDosages(args.out + ".dosages")

if __name__ == "__main__":
    main()