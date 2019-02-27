# Using the python script.
python AlphaFamImpute.py -out example/out -genotypes example/genotypes.txt -pedigree example/pedigree.txt
python AlphaFamImpute.py -out example/phased -genotypes example/genotypes.txt -pedigree example/pedigree.txt -phasefile example/truePhase_parents.txt -extphase
python AlphaFamImpute.py -out example/parent_average -genotypes example/genotypes.txt -pedigree example/pedigree.txt -parentaverage
python AlphaFamImpute.py -out example/em -genotypes example/genotypes.txt -pedigree example/pedigree.txt -em

Rscript example/checkResults.r


# python AlphaFamImpute.py -out test -genotypes example/testGenotypes.txt -pedigree example/testPedigree.txt -em









# # Using the installed program
# # Use e.g., pip install AlphaFamImpute-0.1-py3-none-any.whl

# # Runs the program in "base" mode where full sibs are used to phase the parents. The phased parents are then used to impute the offspring.
# AlphaFamImpute -out out -genotypes example/genotypes.txt -pedigree example/pedigree.txt

# # Runs the program using external phase information on the parents. 
# # This is a best case scenario since true phase information is known.
# AlphaFamImpute -out phased -genotypes example/genotypes.txt -pedigree example/pedigree.txt -phasefile example/truePhase_parents.txt -extphase

# # Runs the program using the parent average genotype for the parents. No linkage, or phase information is used. Genotyping errors and missing genotypes are corrected using single locus peeling.
# # This is a worst case scenario, but may be useful in some applications where no physical map is known.
# AlphaFamImpute -out parent_average -genotypes example/genotypes.txt -pedigree example/pedigree.txt -parentaverage

