# output = as.matrix(read.table("out.dosages"))
# output_pa = as.matrix(read.table("parent_average.dosages"))
# output_phased = as.matrix(read.table("phased.dosages"))
target = as.matrix(read.table("basedata/trueGenotypes.txt"))

getAccuracyFast = function(target, fileName){
    mat = as.matrix(read.table(fileName))
    values = getAccuracy(target, mat)
    print(fileName)
    print(values)
    print(" ")
}

getAccuracy = function(target, mat){
    #Only calculates on the last 80 individuals (these are all low-density offspring)

    indAcc = sapply(23:102, function(i) {
        return(cor(mat[i, -1], target[i, -1]))
    })
    markerAcc = sapply(2:ncol(mat), function(i) {
            return(cor(mat[23:102, i], target[23:102, i]))
    })

    return(c(mean(indAcc), mean(markerAcc, na.rm=T)))
}


getAccuracyFast(target, "outputs/out.dosages")
getAccuracyFast(target, "outputs/parent_average.dosages")
getAccuracyFast(target, "outputs/phased.dosages")
getAccuracyFast(target, "outputs/em.dosages")
