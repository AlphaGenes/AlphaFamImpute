from setuptools import Extension, find_packages, setup

setup(
    name="AlphaFamImpute",
    version="1.0.1",
    author="Andrew Whalen",
    author_email="awhalen@roslin.ed.ac.uk",
    description="An imputation program for phasing and imputation in a full sib family.",
    long_description="An imputation program for phasing and imputation in a full sib family.",
    long_description_content_type="text/markdown",
    url="",
    packages=['alphafamimpute', 'alphafamimpute.FamilyImputation', 'alphafamimpute.tinyhouse'],
    package_dir={'': 'src'},

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points = {
    'console_scripts': [
        'AlphaFamImpute=alphafamimpute.AlphaFamImpute:main'
        ],
    },
    install_requires=[
        'numpy',
        'numba'
    ]
)
