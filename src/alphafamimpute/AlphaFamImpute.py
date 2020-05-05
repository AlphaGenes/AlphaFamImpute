from .tinyhouse import Pedigree
from .tinyhouse import InputOutput
from .tinyhouse import CombinedHMM
from .tinyhouse import ProbMath
from .FamilyImputation import FamilyImputation
from .FamilyImputation import FamilySingleLocusPeeling
# from .FamilyImputation import fsphase # LEGACY
# from .FamilyImputation import FamilyEM # LEGACY
import numpy as np
from numba import jit
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
   
    # Input options
    input_parser = parser.add_argument_group("Input Options")
    InputOutput.add_arguments_from_dictionary(input_parser, InputOutput.get_input_options(), options = ["bfile", "genotypes", "seqfile", "pedigree", "startsnp", "stopsnp", "seed"]) 
    input_parser.add_argument('-map',default=None, required=False, type=str, help='A genetic map file. First column is chromosome name. Second column is basepair position.')

    # Output options
    output_parser = parser.add_argument_group("Output Options")
    output_parser.add_argument('-calling_threshold', default=0.1, required=False, type=float, help='Genotype and phase calling threshold. Default = 0.1 (best guess genotypes).')
    output_parser.add_argument('-supress_genotypes', action='store_true', required=False, help = "Suppresses the output of the called genotypes." )
    output_parser.add_argument('-supress_dosages', action='store_true', required=False, help = "Suppresses the output of the genotype dosages." )
    output_parser.add_argument('-supress_phase', action='store_true', required=False, help = "Suppresses the output of the phase information." )

    # Multithreading
    multithread_parser = parser.add_argument_group("Multithreading Options")
    InputOutput.add_arguments_from_dictionary(multithread_parser, InputOutput.get_multithread_options(), options = ["iothreads"]) 

    # General Arguments
    famParser = parser.add_argument_group("Algorithm Arguments")
    famParser.add_argument('-hd_threshold', default=0.9, required=False, type=float, help='Percentage of non-missing markers to classify an offspring as high-density. Only high-density individuals are used for parent phasing and imputation. Default: 0.9.')
    famParser.add_argument('-gbs', action='store_true', required=False, help='Flag to use all individuals for imputation. Equivalent to: "-hd_threshold 0". Recommended for GBS data.')
    famParser.add_argument('-parentaverage', action='store_true', required=False, help='Runs single locus peeling to impute individuals based on the (imputed) parent-average genotype.')

    InputOutput.add_arguments_from_dictionary(famParser, InputOutput.get_probability_options()) 

    # LEGACY OPTIONS
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
    # TODO: After testing, Move to Tinyhouse (maybe input output).

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

class AlphaFamImputeIndividual(Pedigree.Individual):

    def __init__(self, idx, idn):
        super().__init__(idx, idn)

        self.original_genotypes = None
        self.original_haplotypes = None

    def save_genotypes_and_haplotypes(self):
        self.original_genotypes = self.genotypes.copy()
        self.original_haplotypes = (self.haplotypes[0].copy(), self.haplotypes[1].copy())

    def restore_genotypes_and_haplotypes(self):
        self.genotypes = self.original_genotypes.copy()
        self.haplotypes = (self.original_haplotypes[0].copy(), self.original_haplotypes[1].copy())


class FamilyHMM(CombinedHMM.DiploidMarkovModel):
    def __init__(self, n_loci, error, recombination_rate = None):
        CombinedHMM.DiploidMarkovModel.__init__(self, n_loci, error, recombination_rate)


    def get_point_estimates(self, individual, library_calling_threshold= 0.95, **kwargs):
        paternal_haplotype_library, maternal_haplotype_library, seperate_reference_panels = self.extract_reference_panels(**kwargs)

        paternal_called_haplotypes = paternal_haplotype_library.get_called_haplotypes(threshold = library_calling_threshold)
        maternal_called_haplotypes = maternal_haplotype_library.get_called_haplotypes(threshold = library_calling_threshold)

        mask = self.get_mask(paternal_called_haplotypes) & self.get_mask(maternal_called_haplotypes) 
        
        probs = ProbMath.getGenotypeProbabilities_ind(individual)

        return self.njit_get_point_estimates(probs, paternal_called_haplotypes, maternal_called_haplotypes, self.error, mask)

    @staticmethod
    @jit(nopython=True)
    def njit_get_point_estimates(indProbs, paternalHaplotypes, maternalHaplotypes, error, mask):
        nPat, nLoci = paternalHaplotypes.shape
        nMat, nLoci = maternalHaplotypes.shape

        point_estimates = np.full((nLoci, nPat, nMat), 1, dtype = np.float32)
        for i in range(nLoci):
            if mask[i]:
                for j in range(nPat):
                    for k in range(nMat):
                        # I'm just going to be super explicit here. 
                        p_aa = indProbs[0, i]
                        p_aA = indProbs[1, i]
                        p_Aa = indProbs[2, i]
                        p_AA = indProbs[3, i]
                        e = error[i]

                        # Not super clear why we need this... unless accounting for reference haplotype errors?

                        if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 0:
                            value = p_aa*(1-e)**2 + (p_aA + p_Aa)*e*(1-e) + p_AA*e**2

                        if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 0:
                            value = p_Aa*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_aA*e**2

                        if paternalHaplotypes[j,i] == 0 and maternalHaplotypes[k, i] == 1:
                            value = p_aA*(1-e)**2 + (p_aa + p_AA)*e*(1-e) + p_Aa*e**2

                        if paternalHaplotypes[j,i] == 1 and maternalHaplotypes[k, i] == 1:
                            value = p_AA*(1-e)**2  + (p_aA + p_Aa)*e*(1-e) + p_aa*e**2

                        point_estimates[i,j,k] = value
        return point_estimates




@profile
def main():
    args = getArgs() 
    pedigree = Pedigree.Pedigree(constructor = AlphaFamImputeIndividual) 
    InputOutput.readInPedigreeFromInputs(pedigree, args, genotypes = True, haps = True, reads = True)

    model = FamilyHMM(pedigree.nLoci, args.error, 1/pedigree.nLoci)


    if args.map is None:
        # Run assuming a single chromosome
        run_imputation(pedigree, args, 1/pedigree.nLoci, model)

    else:
        # Split data based on genetic map.
        genetic_map = read_map(args.map)
        for chrom in genetic_map:
            print("Now imputing chromosome", chrom.idx, "using markers", chrom.start, "to", chrom.stop)
            sub_pedigree = pedigree.subset(chrom.start, chrom.stop)
            run_imputation(sub_pedigree, args, chrom.rec_rate, model)
            pedigree.merge(sub_pedigree, chrom.start, chrom.stop)

    if not args.supress_genotypes: pedigree.writeGenotypes(args.out + ".genotypes")
    if not args.supress_dosages: pedigree.writeDosages(args.out + ".dosages")
    if not args.supress_phase: pedigree.writePhase(args.out + ".phase")

def run_imputation(pedigree, args, rec_rate, model):

    model.update_paramaters(pedigree.nLoci, args.error, rec_rate)


    for ind in pedigree:
        ind.save_genotypes_and_haplotypes()

    for fam in pedigree.getFamilies() :
        # Reset parent information
        fam.sire.restore_genotypes_and_haplotypes()
        fam.dam.restore_genotypes_and_haplotypes()

        if args.parentaverage or pedigree.nLoci == 1:
            # Run parent-average genotype.
            FamilySingleLocusPeeling.impute_from_parent_average(fam, pedigree, args)
        
        # elif args.fsphase:
        #     # Legacy fs-phase option.
        #     fsphase.imputeFamUsingFullSibs(fam, pedigree)
        
        # elif args.extphase :
        #     # Legacy option to phase based on a reference panel.
        #     FamilyImputation.imputeFamFromPhasedParents(fam, pedigree)
        
        # elif args.em:
        #     # Legacy option to use an EM algorithm.
        #     FamilyEM.imputeFamUsingFullSibs(fam, pedigree, args)
        
        else:
            # Current, default, phasing algorithm.
            FamilyImputation.impute_family(fam, pedigree, model, rec_rate, args)


if __name__ == "__main__":
    main()