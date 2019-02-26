output = as.matrix(read.table("out.dosages"))
output_pa = as.matrix(read.table("parent_average.dosages"))
output_phased = as.matrix(read.table("phased.dosages"))
target = as.matrix(read.table("example/trueGenotypes.txt"))

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


acc = getAccuracy(target, output)
acc_pa = getAccuracy(target, output_pa)
acc_phased = getAccuracy(target, output_phased)

# > acc
# [1] 0.9933754 0.9798414
# > acc_phased
# [1] 0.9991943 0.9964859
# > acc_pa #The marker-wise accuracy is broken -- all of the children are the same so all of the values are NA.
# [1] 0.899912 1.000000
