options(warn=-1)

previous_run = read.table("last_stable.txt")
print2 = function(string){
    cat(string)
    cat("\n")
}


options(scipen=999)
args = commandArgs(trailingOnly=TRUE)


get_accuracy_file = function(prefix, id_list){
    return(values)
}


get_accuracy = function(target, mat){
    target = as.matrix(target[,-1])
    mat = as.matrix(mat[,-1])
    indAcc = sapply(1:nrow(mat), function(i) {
        return(cor(mat[i,], target[i,]))
    })
    markerAcc = sapply(1:ncol(mat), function(i) {
            return(cor(mat[, i], target[, i]))
    })

    return(c(mean(indAcc), mean(markerAcc, na.rm=T)))
}

get_accuracy_file_folder = function(prefix, folder){


    true_genotypes = read.table(paste0(folder, "/true_genotypes.txt"))
    print("Genotypes")

    id_list = read.table(paste0(folder, "/offspring.txt"))[,1]
    print("Offspring")

    true_offspring = true_genotypes[true_genotypes[,1] %in% id_list,]
    true_parents = true_genotypes[ ! true_genotypes[,1] %in% id_list,]


    imputed_file_name = paste0(folder, "/", prefix, ".dosages")

    imputed = read.table(imputed_file_name)
    imputed_offspring = imputed[imputed[,1] %in% id_list,]
    imputed_parents = imputed[!imputed[,1] %in% id_list,]

    offspring_values = get_accuracy(true_offspring, imputed_offspring)
    parent_values = get_accuracy(true_parents, imputed_parents)

    values = data.frame(generation = c("0", "1"), indAcc = c(parent_values[1], offspring_values[1]), markerAcc = c(parent_values[2], offspring_values[2]))

    previous = previous_run[previous_run$prefix == prefix & previous_run$folder == folder, c("indAcc", "markerAcc")]
    colnames(previous) = c("past_indAcc", "past_markerAcc")

    values = cbind(values, previous)
    rownames(values) = c("parents", "offspring")
    
    print2(paste0("Accuracy on: ", folder, ", ", prefix))
    print(values)
    cat("\n")
    cat("\n")

    values = cbind(folder = folder, prefix = prefix, values)
    run_values[[paste0(prefix, ",", folder)]] <<- values

}

run_values = list()


get_accuracy_file_folder("outputs/out", "example")
get_accuracy_file_folder("outputs/gbs", "example")
get_accuracy_file_folder("outputs/combined", "example")
get_accuracy_file_folder("outputs/parent_average", "example")

get_accuracy_file_folder("outputs/out", "example_map_file")
get_accuracy_file_folder("outputs/gbs", "example_map_file")
get_accuracy_file_folder("outputs/combined", "example_map_file")
get_accuracy_file_folder("outputs/parent_average", "example_map_file")


run_values = do.call("rbind", run_values)
write.table(run_values, "result_summary.txt")
