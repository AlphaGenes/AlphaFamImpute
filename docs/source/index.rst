.. AlphaFamImpute documentation master file, created by
   sphinx-quickstart on Thu Oct 10 10:16:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. NOTE:  added the line to the latex options:   'extraclassoptions': 'openany,oneside'

|program|
==========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. highlight:: none


Introduction
~~~~~~~~~~~~


|program| is a genotype calling, phasing, and imputation algorithm for large full-sib families in diploid plants and animals. |program| supports individuals genotyped with SNP array or GBS data. 

Please report any issues to `John.Hickey@roslin.ed.ac.uk <John.Hickey@roslin.ed.ac.uk>`_ or `awhalen@roslin.ed.ac.uk <awhalen@roslin.ed.ac.uk>`_.

Conditions of use
-----------------
|program| is part of a suite of software that our group has developed. It is fully and freely available for academic use, provided that users cite it in publications. However, due to our contractual obligations with some of the funders of the research program that has developed this suite of software, all interested commercial organizations are requested to contact John Hickey (`John.Hickey@roslin.ed.ac.uk <John.Hickey@roslin.ed.ac.uk>`_) to discuss the terms of use.

Citation and Authorship
-----------------------

|program| is part of a body of imputation software developed by the AlphaGenes group under Professor John Hickey. The approach draws heavily on single and multi-locus iterative peeling (as in AlphaPeel). It also has been inspired by work outside our group such as *hs-phase* (Ferdosi et al. 2014), and *magicimpute* (Zheng et al. 2018). |program| was written by Andrew Whalen, and is currently being supported by Andrew Whalen and Steve Thorn.

Citation:
Whalen, A, Hickey, JM. (2019). *AlphaFamImpute: high accuracy imputation in full-sib families with genotype-by-sequencing data*. ADD BIORXIV INFORMATION.


Disclaimer
----------

While every effort has been made to ensure that |program| does what it claims to do, there is absolutely no guarantee that the results provided are correct. Use of |program| is at your own risk.


Program Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

AlphaFamImpute takes in a number of command line arguments to control the program's behaviour. To view a list of arguments, run AlphaFamImpute without any command line arguments, i.e. ``AlphaFamImpute`` or ``AlphaFamImpute -h``. 


Core Arguments 
--------------

::
  
  Core arguments
    -out prefix              The output file prefix.

The ``-out`` argument gives the output file prefix for where the outputs of AlphaFamImpute should be stored. By default, AlphaFamImpute outputs a file with inputed genotypes, ``prefix.genotypes``, phased haplotypes ``prefix.phase``, and genotype dosages ``prefix.dosages``. For more information on which files are created, see "Output Arguments", below.

Input Arguments 
----------------

::

    Input Options:
      -bfile [BFILE [BFILE ...]]
                          A file in plink (binary) format. Only stable on
                          Linux).
      -genotypes [GENOTYPES [GENOTYPES ...]]
                          A file in AlphaGenes format.
      -seqfile [SEQFILE [SEQFILE ...]]
                          A sequence data file.
      -pedigree [PEDIGREE [PEDIGREE ...]]
                          A pedigree file in AlphaGenes format.
      -startsnp STARTSNP    The first marker to consider. The first marker in the
                          file is marker "1".
      -stopsnp STOPSNP      The last marker to consider.
      -map MAP              A genetic map file. First column is chromosome name.
                            Second column is basepair position.

AlphaFamImpute requires a pedigree file and one or more genotype files to run the analysis.

AlphaFamImpute supports binary plink files, ``-bfile``, genotype files in the AlphaGenesFormat, ``-genotypes``, and sequence data read counts in the AlphaGenes format, ``-seqfile``. A pedigree file must be supplied using the ``-pedigree`` option. 

Use the ``-startsnp`` and ``-stopsnp`` comands to run the analysis only on a subset of markers.

AlphaFamImpute also support using a genetic mapfile, ``-map``, to run analyses across multiple chromosomes. If a map file is not supplied, AlphaFamImpute assumes that the data is all located in a single chromosome. An example map file is provided below.


Output Arguments 
----------------
::

    Output Options:
      -calling_threshold CALLING_THRESHOLD
                            Genotype and phase calling threshold. Default = 0.1
                            (best guess genotypes).
      -supress_genotypes    Suppresses the output of the called genotypes.
      -supress_dosages      Suppresses the output of the genotype dosages.
      -supress_phase        Suppresses the output of the phase information.

By default AlphaFamImpute produces three output files, a genotype file, a phase file, and a genotype dosage file. Creation of each of these files can be suppressed with the ``-supress_genotypes``, ``-supress_dosages``, and ``-supress_phase`` options. 

The ``-calling_threshold`` arguments controls which genotypes (and phased haplotypes) are called as part of the algorithm. A calling threshold of 0.9 indicates that genotypes are only called if greater than 90% of the final probability mass is on that genotype. Using a higher-value will increase the accuracy of called genotypes, but will result in fewer genotypes being called. Since there are three genotypes states,  "best-guess" genotypes are produced with a calling threshold less than ``0.33``. 

Multithreading Arguments 
------------------------
::

    Multithreading Options:
      -iothreads IOTHREADS  Number of threads to use for input and output.
                            Default: 1.

Currently AlphaFamImpute only supports multithreading for reading in and writing out data. The parameter ``-iothreads`` controls the number of threads/processes used by AlphaFamImpute. Setting this option to a value greater than 1 is only recommended for very large files (i.e. >10,000 individuals).

Algorithm Arguments 
------------------------

::

    Algorithm Arguments:
      -hd_threshold HD_THRESHOLD
                            Percentage of non-missing markers to classify an
                            offspring as high-density. Only high-density
                            individuals are used for parent phasing and
                            imputation. Default: 0.9.
      -gbs                  Flag to use all individuals for imputation. Equivalent
                            to: "-hd_threshold 0". Recommended for GBS data.
      -parentaverage        Runs single locus peeling to impute individuals based
                            on the (imputed) parent-average genotype.
      -error ERROR          Genotyping error rate. [Default 0.01]
      -seqerror SEQERROR    Assumed sequencing error rate. [Default 0.001]

As part of the algorithm, AlphaFamImpute breaks the offspring of a full-sib family into a group of high-density and low-density offspring. The high-density offspring are then used to call, phase, and impute the parents. Both the high-density and low-density offspring will be phased and imputed based on the parent's genotypes. The the non-missingness threshold for assigning an individual to the high-density or low-density group is given by the ``-hd_threshold`` argument. The default value of ``0.9`` means that high-density individuals are those where 90% of their markers are not missing; individuals that have more than 10% missing markers are classified as low-density. 

For GBS or sequence data, the recommendation is to include all of the offspring when calling, phasing, and imputing the parents. Use either the ``-gbs`` flag or set ``-hd_threshold 0`` to obtain this behaviour. 

By default, AlphaFamImpute uses an algorithm that assumes that a physical ordering of the markers is known within chromosomes. In cases where physical marker ordering is unknown, AlphaFamImpute can still be used to perform some imputation by estimating the parental genotypes, and using that information as a prior for the offspring's genotypes. use the ``-parentaverage`` option in this case.

The ``-error`` and ``-seqerror`` controlled the assumed genotyping and sequencing error rates. The algorithm is relatively insensitive to the exact values, so we recommend using the default values unless the error rate is known to be much higher than the default. 

Input file formats
~~~~~~~~~~~~~~~~~~

Genotype file 
-------------

Genotype files contain the input genotypes for each individual. The first value in each line is the individual's id. The remaining values are the genotypes of the individual at each locus, either 0, 1, or 2 (or 9 if missing). The following examples gives the genotypes for four individuals genotyped on four markers each.

Example: ::

  id1 0 2 9 0 
  id2 1 1 1 1 
  id3 2 0 2 0 
  id4 0 2 1 0

Sequence file
-------------

The sequence data file is in a similar Sequence data is given in a similar format to the genotype data. For each individual there are two lines. The first line gives the individual's id and the read counts for the reference allele. The second line gives the individual's id and the read counts for the alternative allele.

Example: ::

  id1 4 0 0 7 # Reference allele for id1
  id1 0 3 0 0 # Alternative allele for id2
  id2 1 3 4 3
  id2 1 1 6 2
  id3 0 3 0 1
  id3 5 0 2 0
  id4 2 0 6 7
  id4 0 7 7 0

Pedigree file
-------------

Each line of a pedigree file has three values, the individual's id, their father's id, and their mother's id. "0" represents an unknown id.

Example: ::

  id1 0 0
  id2 0 0
  id3 id1 id2
  id4 id1 id2

Map file
--------

The map file gives the chromosome number and the base pair position of each marker in two columns. Additional columns can be supplied, but will not be used for the program. The following example gives a map file for four markers spread across Chromosome 1 and Chromosome 2. AlphaFamImpute can handle chromosomes with either integer of character names.

Example: ::

  1 12483939
  1 192152913
  2 65429279
  2 107421759

Output file formats
~~~~~~~~~~~~~~~~~~~

Phase file
-----------

The phase file gives the phased haplotypes (either 0 or 1) for each individual in two lines. For individuals where we can determine the haplotype of origin, the first line will provide information on the paternal haplotype, and the second line will provide information on the maternal haplotype.

Example: ::

  id1 0 1 9 0 # Maternal haplotype
  id1 0 1 9 0 # Paternal haplotype
  id2 1 1 1 0
  id2 0 0 0 1
  id3 1 0 1 0
  id3 1 0 1 0 
  id4 0 1 0 0
  id4 0 1 1 0

Dosage file
-----------

The dosage file gives the expected allele dosage for the alternative (or minor) allele for each individual. The first value in each line is the individual ID. The remaining values are the allele dosages at each loci. If there is a lot of uncertainty in the underlying genotype calls, this file can often prove more accurate than the called genotype values.

Example: ::

  1    0.0003    2.0000    1.0000    0.0001
  2    1.0000    1.0000    1.0000    1.0000
  3    2.0003    0.0000    2.0000    0.0001
  4    0.0000    2.0000    1.0000    0.0000

.. |program| replace:: AlphaFamImpute