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
    famParser.add_argument('-hd_threshold', default=0.9, required=False, type=float, help='Percentage of non-missing markers to classify an offspring as high-density. Only high-density individuals are used for parent phasing and imputation. Default: 0.9.')
    famParser.add_argument('-gbs', action='store_true', required=False, help='Flag to use all individuals for imputation. Equivalent to: "-hd_threshold 0". Recommended for GBS data.')
    famParser.add_argument('-parentaverage', action='store_true', required=False, help='Runs single locus peeling to impute individuals based on the (imputed) parent-average genotype.')

    output_parser = parser.add_argument_group("AlphaFamImpute Output Options")

    output_parser.add_argument('-calling_threshold', default=0.1, required=False, type=float, help='Genotype and phase calling threshold. Default = 0.1 (best guess genotypes).')
    output_parser.add_argument('-supress_genotypes', action='store_true', required=False, help = "Supresses the output of the called genotypes." )
    output_parser.add_argument('-supress_dosages', action='store_true', required=False, help = "Supresses the output of the genotype dosages." )
    output_parser.add_argument('-supress_phase', action='store_true', required=False, help = "Supresses the output of the phase information." )

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

    # Split data based on genetic map.

    if args.map is None:
        run_imputation(pedigree, args, 1/pedigree.nLoci)

    else:
        genetic_map = read_map(args.map)
        for chrom in genetic_map:
            print("Now imputing chromosome", chrom.idx)
            sub_pedigree = pedigree.subset(chrom.start, chrom.stop)
            run_imputation(sub_pedigree, args, chrom.rec_rate)
            pedigree.merge(sub_pedigree, chrom.start, chrom.stop)

    if not args.supress_genotypes: pedigree.writeGenotypes(args.out + ".genotypes")
    if not args.supress_dosages: pedigree.writeDosages(args.out + ".dosages")
    if not args.supress_phase: pedigree.writePhase(args.out + ".phase")

def run_imputation(pedigree, args, rec_rate):
    for fam in pedigree.getFamilies() :
        if args.parentaverage or pedigree.nLoci == 1:
            # Default to parent-average genotype in these situations.
            FamilySingleLocusPeeling.imputeFamFromParentAverage(fam, pedigree, args)
        
        elif args.fsphase:
            # Legacy fs-phase option.
            fsphase.imputeFamUsingFullSibs(fam, pedigree)
        
        elif args.extphase :
            # Legacy option to phase based on a reference panel.
            FamilyImputation.imputeFamFromPhasedParents(fam, pedigree)
        
        elif args.em:
            # Legacy option to use an EM algorithm.
            FamilyEM.imputeFamUsingFullSibs(fam, pedigree, args)
        
        else:
            # Current, default, phasing algorithm.
            FamilyImputation.impute_family(fam, pedigree, rec_rate, args)


if __name__ == "__main__":
    main()