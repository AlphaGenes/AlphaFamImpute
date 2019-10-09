

python ../src/AlphaFamImpute-script.py -seqfile sequence.txt -pedigree pedigree.txt -out outputs/magic; Rscript checkResults.r


# kernprof -l ../src/AlphaFamImpute-script.py -seqfile sequence.txt -pedigree pedigree.txt -out outputs/magic -magic
# python -m line_profiler AlphaFamImpute-script.py.lprof > tmp.txt; open tmp.txt



# # options(scipen = 999); map = data.frame(rep(c("1", "2"), each = 500), 1:1000 * (1000000*100/1000)); write.table(map, "map.txt", row.names=F, col.names = F, quote = F)
# options(scipen = 999); map = data.frame("1", 1:1000 * (1000000*100/1000)); write.table(map, "map.txt", row.names=F, col.names = F, quote = F)