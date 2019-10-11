# # Using the installed program
# # Use e.g., pip install AlphaFamImpute-0.1-py3-none-any.whl


########################
## Basic run commands ##
########################

# # Runs the program in "base" mode where high-density full sibs are used to phase the parents. The phased parents are then used to impute the offspring.
AlphaFamImpute -out example/out -genotypes example/genotypes.txt -pedigree example/pedigree.txt

# # Runs the program using GBS data to call and phase the parents and offspring.
# # The "-gbs" flag is used to indicate that all individuals should be used to phase the parents. 
AlphaFamImpute -out example/gbs -seqfile example/sequence.txt -pedigree example/pedigree.txt -gbs

# # Runs the program using the parent average genotype for the parents using single locus peeling. No linkage, or phase information is used.
# # Run only if marker ordering is unknown.
AlphaFamImpute -out example/parent_average -seqfile example/sequence.txt -pedigree example/pedigree.txt -parentaverage


###############
## Map files ##
###############

# # A map file can also be used to impute multiple chromosomes at the same time. 
# # If a map file is not supplied, it is assumed that the program is given a single chromosome. 

AlphaFamImpute -out example_map_file/out -seqfile example_map_file/sequence.txt -pedigree example_map_file/pedigree.txt -gbs -map example_map_file/map.txt
