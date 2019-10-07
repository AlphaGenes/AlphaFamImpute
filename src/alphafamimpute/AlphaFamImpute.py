from .tinyhouse import Pedigree
from .tinyhouse import InputOutput
from .FamilyImputation import FamilyImputation
from .FamilyImputation import FamilySingleLocusPeeling
from .FamilyImputation import fsphase
from .FamilyImputation import FamilyEM
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
    InputOutput.add_genotype_probability_arguments(parser)
    
    famParser = parser.add_argument_group("AlphaFamImpute Arguments")
    famParser.add_argument('-map',default=None, required=False, type=str, help='A genetic map file. First column is chromosome name. Second column is basepair position.')
    famParser.add_argument('-hd_threshold', default=0.9, required=False, type=float, help='Threshold to include high-density individuals for imputation. Default: 0.9.')
    famParser.add_argument('-gbs', action='store_true', required=False, help='Flag to use all individuals for imputation. Equivalent to: "-hd_threshold 0". Recommended for calling GBS data.')
    famParser.add_argument('-parentaverage', action='store_true', required=False, help='Runs single locus peeling to impute offspring. Use if a map-file is unavailable. ')

    # Depreciated run option. 
    famParser.add_argument('-extphase', action='store_true', required=False, help=argparse.SUPPRESS) #  help='Include when using an external phase file for the parents. Skips the fullsib phase algorithm and just runs the HMM.')
    famParser.add_argument('-fsphase', action='store_true', required=False, help=argparse.SUPPRESS) # 'Flag to run a full-sib phasing algorithm.')

    # Old arguments for the EM algorithm. Now depreciated. 
    famParser.add_argument('-em', action='store_true', required=False, help = argparse.SUPPRESS) # help='Include to impute based on parent average genotype. This runs single locus peeling on the parents to fill in any missing genotypes, then peels down to each offspring.')

    famParser.add_argument('-ncycles', default=30, required=False, type=int, help=argparse.SUPPRESS) # help='Number of EM cycles. Only has an effect with EM mode, Default = 30.')
    famParser.add_argument('-niter', default=10, required=False, type=int, help=argparse.SUPPRESS) #help='Number of full iterations. Only has an effect with EM mode, Default = 10.')
    famParser.add_argument('-jitter', default=0.01, required=False, type=float, help=argparse.SUPPRESS) #help='Jitter amount for the initial haplotypes. Only has an effect with EM mode. Default = 0.01.')

    famParser.add_argument('-usegenoprobs', action='store_true', required=False, help=argparse.SUPPRESS) #help='Flag to use the individual\'s genotype probabilities for calculating dosages.')
    famParser.add_argument('-preimpute', action='store_true', required=False, help=argparse.SUPPRESS), #help='Flag to use the children\'s genotypes to pre-impute the parent\'s genotypes (via single locus peeling).')

    return InputOutput.parseArgs("AlphaFamImpute", parser)


class Chromosome(object) :
    def __init__(self, idx, loci_range, bp_range):
        print(idx, loci_range, bp_range)
        self.idx = idx
        self.start, self.stop = loci_range

        self.nLoci = self.stop - self.start

        size_in_mbp = (bp_range[1] - bp_range[0]) / 1000000.0
        size_in_cm = 1*size_in_mbp
        size_in_m = size_in_cm/100

        if self.nLoci > 1:
            self.rec_rate = size_in_m/(self.nLoci-1) # Assume evenly split up between markers.
        else:
            self.rec_rate = 0

def read_map(file_name):
    genetic_map = []
    with open(file_name) as f:
        lines = f.readlines()

    current_chr = None
    start = 0
    bp_start = 0
    bp_last = 0
    for i, line in enumerate(lines):
        parts = line.split()
        idx = parts[0]; 
        bp = int(parts[1])
        
        if current_chr is None:
            current_chr = idx
            start = 0
            bp_start = bp

        if current_chr != idx:
            new_chr = Chromosome(current_chr, (start, i), (bp_start, bp_last))
            genetic_map.append(new_chr)

            current_chr = idx
            start = i
            bp_start = bp

        bp_last = bp

    new_chr = Chromosome(current_chr, (start, len(lines)), (bp_start, bp_last))
    genetic_map.append(new_chr)

    return genetic_map


@profile
def main():
    args = getArgs() 
    pedigree = Pedigree.Pedigree() 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True, reads = True)

    if args.map is None:
        run_imputation(pedigree, args, 1/pedigree.nLoci)

    else:
        genetic_map = read_map(args.map)
        for chrom in genetic_map:
            print(chrom.start, chrom.stop, chrom.rec_rate)
            sub_pedigree = pedigree.subset(chrom.start, chrom.stop)
            run_imputation(sub_pedigree, args, chrom.rec_rate)
            pedigree.merge(sub_pedigree, chrom.start, chrom.stop)

    pedigree.writePhase(args.out + ".phase")
    pedigree.writeGenotypes(args.out + ".genotypes")
    pedigree.writeDosages(args.out + ".dosages")

def run_imputation(pedigree, args, rec_rate):
    for fam in pedigree.getFamilies() :
        if args.parentaverage or pedigree.nLoci == 1:
            FamilySingleLocusPeeling.imputeFamFromParentAverage(fam, pedigree)
        
        elif args.fsphase:
            fsphase.imputeFamUsingFullSibs(fam, pedigree)
        
        elif args.extphase :
            FamilyImputation.imputeFamFromPhasedParents(fam, pedigree)
        
        elif args.em:
            FamilyEM.imputeFamUsingFullSibs(fam, pedigree, args)
        
        else:
            FamilyImputation.imputeFamUsingFullSibs(fam, pedigree, rec_rate, args)


if __name__ == "__main__":
    main()