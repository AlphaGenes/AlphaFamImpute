python AlphaFamImpute.py -out out -genotypes example/genotypes.txt -pedigree example/pedigree.txt
python AlphaFamImpute.py -out phased -genotypes example/genotypes.txt -pedigree example/pedigree.txt -phasefile example/truePhase_parents.txt -extphase
python AlphaFamImpute.py -out parent_average -genotypes example/genotypes.txt -pedigree example/pedigree.txt -parentaverage

