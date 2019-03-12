# # Using the installed program
# # Use e.g., pip install AlphaFamImpute-0.1-py3-none-any.whl

# # Runs the program in "base" mode where full sibs are used to phase the parents. The phased parents are then used to impute the offspring.
AlphaFamImpute -out outputs/out -genotypes data/genotypes.txt -pedigree data/pedigree.txt

# # Runs the program using an EM algorithm to phase the parental genotypes. These genotypes are then used to impute the offspring.
AlphaFamImpute -out outputs/EM -genotypes data/genotypes.txt -pedigree data/pedigree.txt -em

# # Runs the program using external phase information on the parents. 
# # This is a best case scenario since true phase information is known.
AlphaFamImpute -out outputs/phased -genotypes data/genotypes.txt -pedigree data/pedigree.txt -phasefile data/truePhase_parents.txt -extphase

# # Runs the program using the parent average genotype for the parents. No linkage, or phase information is used. Genotyping errors and missing genotypes are corrected using single locus peeling.
# # This is a worst case scenario, but may be useful in some applications where no physical map is known.
AlphaFamImpute -out outputs/parent_average -genotypes data/genotypes.txt -pedigree data/pedigree.txt -parentaverage

