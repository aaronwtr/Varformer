# Genetic Drug Targeting and Mutant Allele Classification
### Project aim
In order to prioritize drug targets, particular mutant alleles, which could be loss-of-function (LoF) or gain-of-function (GoF) mutations, should be classified on whether they are either benign or malignant. Furthermore, for the malignant mutant alleles, we can perform downstream prioritization by looking at clinical trial outcome data for compounds targeting that allele. 
### Data
- DrugBank: Clinical trial data targeting diseased pathways/targets 

- ELGH: Around 50,000 whole exomes of primarily Bangladeshi and Pakistani population. Also contains electronic health records.

- UK BioBank: Database tracking health of 500,000 participants. Note that UKBB is not representative of general population. Skew towards population of Caucasian descent between the ages of 40-69. 
### Methods 
- Mutant allele classification: For the LoF and GoF classification, we will build a simple logistic regression model taking inspiration from previous work at the Centre for Translational Bioinformatics. Logistic regression is an intuitive first step as this architecture is simple to construct, should inform us whether any meaningful prediction can be done, and allows for binary classification via a threshold, e.g. risk > 50% = malignant mutation, else benign. Moreover, the risk probabilities can be used to prioritize the targets in rank order, where the targets with the biggest malignant risk would be the most obvious targets. 

- Clinical trial outcome-based drug target prioritization: T.B.D
