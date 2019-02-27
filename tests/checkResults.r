output = as.matrix(read.table("out.dosages"))
target = as.matrix(read.table("trueGenotypes.txt"))

indAcc = sapply(23:102, function(i) {
            return(cor(output[i, -1], target[i, -1]))
    })

markerAcc = sapply(2:ncol(output), function(i) {
            return(cor(output[23:102, i], target[23:102, i]))
    })

