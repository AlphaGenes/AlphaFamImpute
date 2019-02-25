from tinyhouse import Pedigree
from tinyhouse import InputOutput
from FamilyImputation import FamilyImputation
import numpy as np

try:
    profile
except:
    def profile(x): 
        return x

@profile
def main():
    
    args = InputOutput.parseArgs("AlphaCall")
    pedigree = Pedigree.Pedigree() 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True, reads = True)

    for fam in pedigree.getFamilies() :
        FamilyImputation.imputeFam(fam, pedigree)

    pedigree.writePhase(args.out + ".phase")
    pedigree.writeGenotypes(args.out + ".genotypes")
    pedigree.writeDosages(args.out + ".dosages")

if __name__ == "__main__":
    main()