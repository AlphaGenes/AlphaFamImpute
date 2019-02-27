#Uses devel version of AlphaSimR '0.3.0'
# > devtools::install_bitbucket("hickeyjohnteam/AlphaSimR", args = '--library=.')
library(AlphaSimR)

options(scipen=999)
args = commandArgs(trailingOnly=TRUE)

cond = args[1]

# print(args)
founderPop = runMacs(nInd = 100, nChr = 1, segSites = 1000)

# ---- Set Simulation Parameters ----

SP = SimParam$new(founderPop)
SP$addTraitA(nQtlPerChr=1000, mean=0, var=1)
SP$addSnpChip(nSnpPerChr=1000)
SP$setVarE(h2=.4)

gen0 = newPop(founderPop)

gen1 = selectCross(gen0, nInd = 100, nCrosses = 2, nProgeny = 1)
gen2 = selectCross(gen1, nInd = 2, nCrosses = 1, nProgeny = 100)
output = c(gen1, gen2) 

write.table2 = function(x, file, ...) {
    write.table(x, file, col.names=F, row.names=F,quote=F,...)
}


pedigree = data.frame(output@id, output@father, output@mother, stringsAsFactors = FALSE)
genotypes = AlphaSimR::pullSnpGeno(output, 1)
haplotypes = AlphaSimR::pullSnpHaplo(output, 1)

indexes = lapply(1:nrow(genotypes), function(index) {
    val0 = (index -1)*2 + 1
    val1 = (index -1)*2 + 2
    return(c(val1, val0))
})
indexes = do.call("c", indexes)
haplotypes = haplotypes[indexes,]

pedigree[1:2, 2:3] = 0
write.table2(pedigree, "pedigree.txt")

write.table2(cbind(pedigree[,1], genotypes), "trueGenotypes.txt")
write.table2(cbind(rep(pedigree[,1], each = 2), haplotypes), "truePhase.txt")

# Grab last ~ 80 individuals.
maskedGenotypes = genotypes

errors = which(rbinom(length(maskedGenotypes), 1, .015) == 1)
newGenotype = sample(c(0, 1, 2), length(errors), replace =T)
maskedGenotypes[errors] = newGenotype

ldInd = c(1:2, 6:102)

ldLoci = seq(1, 1000, by = 20)
maskedLoci = setdiff(1:1000, ldLoci)
maskedGenotypes[ldInd, maskedLoci] = 9



write.table2(cbind(pedigree[,1], maskedGenotypes), "genotypes.txt")

