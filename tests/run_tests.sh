# AlphaFamImpute=AlphaFamImpute
AlphaFamImpute='python ../src/AlphaFamImpute-script.py'

function program_run_message {
    echo ""
    echo ""
    echo Now running $1
    echo ""
}

########################
##     no map file    ##
########################

rm -r example/outputs
mkdir example/outputs


program_run_message Genotypes
$AlphaFamImpute -out example/outputs/out -genotypes example/masked_genotypes.txt -pedigree example/pedigree.txt -seed 12345

program_run_message GBS
$AlphaFamImpute -out example/outputs/gbs -seqfile example/sequence.txt -pedigree example/pedigree.txt -gbs -seed 12345

program_run_message Genotypes + GBS
$AlphaFamImpute -out example/outputs/combined -seqfile example/sequence.txt -genotypes example/masked_genotypes.txt -pedigree example/pedigree.txt -gbs -seed 12345

program_run_message Parent Average
$AlphaFamImpute -out example/outputs/parent_average -seqfile example/sequence.txt -pedigree example/pedigree.txt -parentaverage -seed 12345


########################
##   with map file    ##
########################

rm -r example_map_file/outputs
mkdir example_map_file/outputs

program_run_message Genotypes with map file
$AlphaFamImpute -out example_map_file/outputs/out -genotypes example_map_file/masked_genotypes.txt -pedigree example_map_file/pedigree.txt -seed 12345 -map example_map_file/map.txt

program_run_message GBS with map file
$AlphaFamImpute -out example_map_file/outputs/gbs -seqfile example_map_file/sequence.txt -pedigree example_map_file/pedigree.txt -gbs -seed 12345 -map example_map_file/map.txt

program_run_message Genotypes + GBS with map file
$AlphaFamImpute -out example_map_file/outputs/combined -seqfile example_map_file/sequence.txt -genotypes example_map_file/masked_genotypes.txt -pedigree example_map_file/pedigree.txt -gbs -seed 12345 -map example_map_file/map.txt

program_run_message Parent Average with map file
$AlphaFamImpute -out example_map_file/outputs/parent_average -seqfile example_map_file/sequence.txt -pedigree example_map_file/pedigree.txt -parentaverage -seed 12345 -map example_map_file/map.txt

Rscript check_results.r