
# Create python wheel.
rm -r build
rm -r dist
python setup.py bdist_wheel

#Compile manual
 ( cd docs; make latexpdf )


target=AlphaFamImpute
rm -rf $target
mkdir $target
cp dist/* $target
cp docs/build/latex/AlphaFamImpute.pdf $target
cp -r example/* $target
zip -r $target.zip AlphaFamImpute

