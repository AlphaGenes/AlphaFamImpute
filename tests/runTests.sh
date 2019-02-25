rm outputs/*
python ../AlphaCall.py -out outputs/out -seqfile basedata/reads.txt -threshold 0.98

Rscript checkResults.r