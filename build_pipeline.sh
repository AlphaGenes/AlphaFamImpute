rm -r build
rm -r dist
python setup.py bdist_wheel
pip uninstall AlphaFamImpute -y
pip install dist/AlphaFamImpute-0.1-py3-none-any.whl


rm -rf AlphaFamImpute_dist
mkdir AlphaFamImpute_dist
cp dist/* AlphaFamImpute_dist
cp -r example AlphaFamImpute_dist
zip -r AlphaFamImpute.zip AlphaFamImpute_dist

