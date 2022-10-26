# Genetic Drug Targeting and Mutant Allele Classification
### Project aim
In order to prioritize drug targets, particular mutant alleles, which could be loss-of-function (LoF) or gain-of-function (GoF) mutations, should be classified on whether they are either benign or pathogenic. Furthermore, for the pathogenic mutant alleles, we can perform downstream prioritization by looking at clinical trial outcome data for compounds targeting that allele. 

### Methods 
- Mutant allele classification: For the LoF and GoF classification, we will build a simple logistic regression model taking inspiration from previous work at the Centre for Translational Bioinformatics. Logistic regression is an intuitive first step as this architecture is simple to construct, should inform us whether any meaningful prediction can be done, and allows for binary classification via a threshold, e.g. risk > 50% = pathogenic mutation, else benign. Moreover, the risk probabilities can be used to prioritize the variant alleles in rank order, where the alleles with the biggest pathogenic risk would be the most obvious targets. Alternatively, previous work by Daniel Rhodes showed via TPOT-mediated model selection that random forest models perform best for this task. This could be a starting point and eventual baseline as well. 

- Clinical trial outcome-based drug target prioritization: T.B.D
### Data
- Genomic data
  - East London Genes and Health (ELGH) contains around 50,000 whole exomes of primarily Bangladeshi and Pakistani populations. Also contains electronic health records. https://www.genesandhealth.org/research/scientific-data-downloads
  - UK BioBank (UKBB) is a database that tracks the health of 500,000 participants, including genetic screens. Note that UKBB is not representative of general population. Skew towards population of Caucasian descent between the ages of 40-69.  https://www.ukbiobank.ac.uk/
  - Born in Bradford is a consortium tracking the health of about 13,000 participants from Bradford. Data includes EHR and genomes. https://borninbradford.github.io/datadict/bib/bib_biobank/current_samples/
  - deCODE is a research company that performed whole-genome sequencing experiments for about 2,500 individuals. Couldn't find open access to this dataset online. https://www.decode.com/
  - PROMIS (Patient-Reported Outcomes Measurement Information System) is a set of person-centred measures that evaluates and monitors physical, mental, and social health in adults and children. It can be used with the general population and with individuals living with chronic conditions. Also performed genotyping but could not immediately find this data. https://www.healthmeasures.net/explore-measurement-systems/promis 
  - ClinVar is a freely accessible, public archive of reports on the relationships among human variations and phenotypes, with supporting evidence. ClinVar thus facilitates access to and communication about the relationships asserted between human variation and observed health status, and the history of that interpretation. This resource can be used to assess e.g. haploinsufficiency. https://www.ncbi.nlm.nih.gov/clinvar/
  - OMIM is a comprehensive, authoritative compendium of human genes and genetic phenotypes that is freely available and updated daily. The full-text, referenced overviews in OMIM contain information on all known mendelian disorders and over 16,000 genes. OMIM focuses on the relationship between phenotype and genotype. https://www.omim.org/

- Clinical trial status data for drug targets 
  - DrugBank aggregates clinical trial status data for particular targets in a structured fashion, e.g. https://go.drugbank.com/drugs/DB00001. Not immediately clear whether this data is also queryable in this structured way via API. 
  - ClinicalTrials.gov has an API that is queryable. However, organisation seems unstructured compared to DrugBank. https://clinicaltrials.gov/api/gui
  - DGIdb has a druggability assessment tool and drug-gene interaction search engine. The database is queryable through API. https://www.dgidb.org/api
  - Open Targets Genetics is a comprehensive tool highlighting variant-centric statistical evidence to allow both prioritisations of candidate causal variants at trait-associated loci and identification of potential drug targets. Accessible via GraphQL API. https://genetics-docs.opentargets.org/data-access/graphql-api
