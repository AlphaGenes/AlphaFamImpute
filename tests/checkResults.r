options(scipen=999)
args = commandArgs(trailingOnly=TRUE)


true_genotypes = as.matrix(read.table("trueGenotypes.txt"))
idList = read.table("offspring.txt")[,1]

true_offspring = true_genotypes[true_genotypes[,1] %in% idList,]
true_parents = true_genotypes[ ! true_genotypes[,1] %in% idList,]


getAccuracyFast = function(prefix){
    out = tryCatch({
        fileName = paste0(prefix, ".dosages")

        imputed = as.matrix(read.table(fileName))
        imputed_offspring = imputed[imputed[,1] %in% idList,]
        imputed_parents = imputed[!imputed[,1] %in% idList,]

        offspring_values = getAccuracy(true_offspring, imputed_offspring)
        parent_values = getAccuracy(true_parents, imputed_parents)

        values = data.frame(algorithm = prefix, generation = c("0", "1"), indAcc = c(parent_values[1], offspring_values[1]), markerAcc = c(parent_values[2], offspring_values[2]))
        return(values)
    },
    error=function(cond){
        return(NULL)
    })
}


getAccuracy = function(target, mat){

    indAcc = sapply(1:nrow(mat), function(i) {
        return(cor(mat[i, -1], target[i, -1]))
    })
    markerAcc = sapply(2:ncol(mat), function(i) {
            return(cor(mat[, i], target[, i]))
    })

    return(c(mean(indAcc), mean(markerAcc, na.rm=T)))
}
# getAccuracyFast("em")


values = list(  getAccuracyFast("outputs/magic") )
values = do.call("rbind", values)
print(values)



prefix = "outputs/magic"
fileName = paste0(prefix, ".dosages")

imputed = as.matrix(read.table(fileName))
imputed_offspring = imputed[imputed[,1] %in% idList,]
imputed_parents = imputed[!imputed[,1] %in% idList,]

indAcc = sapply(1:nrow(imputed_offspring), function(i) {
    return(cor(imputed_offspring[i, -1], true_offspring[i, -1]))
})
markerAcc = sapply(2:ncol(imputed_offspring), function(i) {
        return(cor(imputed_offspring[, i], true_offspring[, i]))
})


plot(indAcc)
plot(markerAcc)

