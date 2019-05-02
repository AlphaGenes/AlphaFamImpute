mkdir outputs
rm outputs/*

# Using the python script.
AlphaFamImpute -out outputs/out -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt
AlphaFamImpute -out outputs/phased -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -phasefile basedata/truePhase_parents.txt -extphase
AlphaFamImpute -out outputs/parent_average -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -parentaverage
AlphaFamImpute -out outputs/em -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -em

Rscript checkResults.r
