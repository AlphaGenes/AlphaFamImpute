mkdir outputs
rm outputs/*

# Using the python script.
python ../AlphaFamImpute.py -out outputs/out -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt
python ../AlphaFamImpute.py -out outputs/phased -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -phasefile basedata/truePhase_parents.txt -extphase
python ../AlphaFamImpute.py -out outputs/parent_average -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -parentaverage
python ../AlphaFamImpute.py -out outputs/em -genotypes basedata/genotypes.txt -pedigree basedata/pedigree.txt -em

Rscript checkResults.r
