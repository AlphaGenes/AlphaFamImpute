from tinyhouse import Pedigree
from tinyhouse import InputOutput
from FamilyImputation import FamilyImputation
from FamilyImputation import FamilySingleLocusPeeling
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

    return InputOutput.parseArgs("AlphaFamImpute", parser)


@profile
def main():
    args = getArgs() 
    pedigree = Pedigree.Pedigree() 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True, reads = True)

    for fam in pedigree.getFamilies() :
        if args.extphase :
            FamilyImputation.imputeFamFromPhasedParents(fam, pedigree)
        elif args.parentaverage:
            FamilySingleLocusPeeling.imputeFamFromParentAverage(fam, pedigree)
        else:
            FamilyImputation.imputeFamUsingFullSibs(fam, pedigree)

    pedigree.writePhase(args.out + ".phase")
    pedigree.writeGenotypes(args.out + ".genotypes")
    pedigree.writeDosages(args.out + ".dosages")

if __name__ == "__main__":
    main()