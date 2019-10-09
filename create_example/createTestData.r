#Uses devel version of AlphaSimR '0.3.0'
# > devtools::install_bitbucket("hickeyjohnteam/AlphaSimR", args = '--library=.')
library(AlphaSimR)

options(scipen=999)
args = commandArgs(trailingOnly=TRUE)
args = c("200", "1", "GBS")
nHD = as.numeric(args[1])
coverage = as.numeric(args[2])
parentStatus = args[3]

# print(args)
founderPop = runMacs(nInd = 100, nChr = 1, segSites = 200, inbred=FALSE, species="CATTLE")
# founderPop = runMacs(nInd = 100, nChr = 1, segSites = 1000, inbred=FALSE)

# ---- Set Simulation Parameters ----

SP = SimParam$new(founderPop)
SP$addSnpChip(nSnpPerChr=200)

gen0 = newPop(founderPop)

gen_parents = randCross(gen0, nCrosses = 10, nProgeny = 1)
ids = gen_parents@id

fathers = rep(1:5, each = 20)
mothers = rep(6:10, each = 20)

cross_plan = data.frame(fathers, mothers)

gen_offspring = makeCross(gen_parents, cross_plan)
output = c(gen_parents, gen_offspring) 

write.table2 = function(x, file, ...) {
    write.table(x, paste0("example/", file), col.names=F, row.names=F,quote=F,...)
}

# PEDIGREE

ped_parents = data.frame(id = paste0("ind_", gen_parents@id), father = 0, mother = 0, stringsAsFactors = FALSE)
ped_offspring = data.frame(id = paste0("ind_", gen_offspring@id), father = paste0("ind_", gen_offspring@father), mother = paste0("ind_", gen_offspring@mother), stringsAsFactors = FALSE)
pedigree = rbind(ped_parents, ped_offspring)
write.table2(pedigree, "pedigree.txt")

# GENOTYPES

genotypes_parents = AlphaSimR::pullSnpGeno(gen_parents, 1)
genotypes_offspring = AlphaSimR::pullSnpGeno(gen_offspring, 1)
genotypes = AlphaSimR::pullSnpGeno(output, 1)
write.table2(cbind(pedigree[,1], genotypes), "genotypes.txt")

# HAPLOTYPES

haplotypes = AlphaSimR::pullSnpHaplo(output, 1)

# Swap the haplotype values to be consistent with other software
indexes = lapply(1:nrow(genotypes), function(index) {
    val0 = (index -1)*2 + 1
    val1 = (index -1)*2 + 2
    return(c(val1, val0))
})
indexes = do.call("c", indexes)
haplotypes = haplotypes[indexes,]


# write.table2(cbind(rep(pedigree[,1], each = 2), haplotypes), "truePhase.txt")
# write.table2(cbind(rep(pedigree[,1], each = 2), haplotypes)[1:4,], "truePhase-parents.txt")


gbsRandom = function(id, genotype, coverage) {
    nSnp = length(genotype)

    counts = rpois(nSnp, coverage)
    error = 0.001
    probs = (1-error) * as.numeric(genotype)/2.0 + error/2
    
    altReads = rbinom(nSnp, counts, prob = probs)
    values = matrix(c(id, counts-altReads, id, altReads), nrow = 2, byrow=TRUE)

    if(length(sequence) == 0) {
        sequence <<- values
    }
    else{
        sequence <<- rbind(sequence, values)
    }
}

sequence = c()

generate_sequence_data = function(pedigree, genotypes, coverage) {

    nInd = nrow(pedigree)
    for(i in 1:nInd){
        gbsRandom(pedigree[i, 1], genotypes[i,], coverage)
    }

}

if(parentStatus == "GBS") {
    generate_sequence_data(ped_parents, genotypes_parents, coverage)
}

if(parentStatus == "HD") {
    generate_sequence_data(ped_parents, genotypes_parents, 25)
}
if(parentStatus == "ND") {
    generate_sequence_data(ped_parents, genotypes_parents, 0)
}

generate_sequence_data(ped_offspring, genotypes_offspring, coverage)

write.table2(sequence, "sequence.txt")
