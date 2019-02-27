python setup.py bdist_wheel

rm -rf AlphaFamImpute_dist
mkdir AlphaFamImpute_dist
cp dist/* AlphaFamImpute_dist
cp -r example AlphaFamImpute_dist
zip -r AlphaFamImpute.zip AlphaFamImpute_dist

