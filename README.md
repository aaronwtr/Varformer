# Genetic Drug Targeting and Mutant Allele Classification
### Project aim
This project is separated into two related research: 

1) Genomic drug target prediction (GDT) 
2) Variant pathogenicity and drug target prediction (VaPaD)

These two are not mutually exclusive but rather set up in parallel. Most of the methods and ideas for the GDT project should be transferrable to the VaPaD project. Hence, in the end, these two projects should be reconciled into one drug target prediction model that incorporates variant pathogenicity.

### Genomic Drug Target Prediction
For the GDT project, the idea is to build a database for each variant of interest including, but not limited to, the following data: 

| Data type                                 |Description                                                                          |
|-------------------------------------------|-------------------------------------------------------------------------------------|
| <em>Druggability </em>                    | <em> Calculate a druggability score (are there ground truth labels for this?) </em> |
| Drug status                               | Has the gene been drugged yes/no. If yes, what was the clinical outcome?            |
| Target efficacy score                     | Calculate how efficacious the target is (using clinical trial data for the time being? Doing this calculation based on structural information would be more causal but might not yet be as feasible)                                            |
| Target safety score                       | Calculate safety score for the target. Can use black box warnings and/or existing toxicity measures from e.g. DrugBank.                                                                                                                         |
| Target conservation                       | Metric indicating the extent to which the particular target is conserved from an evolutionary perspective (dN/dS)                                                                                                                           |
| Expression levels                         | Quantification of gene expression (RNA-seq). For this to make sense we need to narrow in on particular cell types or tissues.                                                                                                                 |
| Stringency of gene regulatory sequences   | Regulatory sequences play a big role in gene expression and so defects in stringent gene regulatory sequences can have pathogenic consequences. How to go about quantifying this?                                                               |

This data will be used to build an ML model to that predicts <em>druggability (target variable)</em>. This means ground truth labels for druggability are needed. We can start building and training this model on pre-existing datasets without taking pathogenicity into account. In the end, we can reconcile this model with the pathogenicity prediction to find high quality (i.e. high druggability score) and high impact (i.e. highly pathogenic) targets for all structural variants. The outlined dataset can be built from the sources in [Data](#data). 

### Variant pathogenicity and drug target prediction
Predicting pathogenicity of a particular target is important to identify high impact targets, i.e. genes that are causally involved in severe disease indications. With recent advances in large language and generative AI models, now seems like an ideal time to incorporate sequence-based pathogenicity predictions into the drug discovery pipeline. In the end, sequence-based pathogenicty prediction can alleviate sparse and unreliable pathogenicity annotations in existing databases as they are already showing tremendous ability to generalize, e.g. [https://pubmed.ncbi.nlm.nih.gov/34707284/](https://pubmed.ncbi.nlm.nih.gov/34707284/) and [https://www.biorxiv.org/content/10.1101/2022.08.25.505311v1](https://www.biorxiv.org/content/10.1101/2022.08.25.505311v1). When successfull, sequence-based pathogenicity will be an important measure in guiding drug target discovery and druggability because we want to maximize both the target quality, i.e. can the proposed target be succesfully targeted, and target impact, i.e. is this target associated with a diseased phenotype? 

<b>{To be worked out in more detail in collaboration with W. Lin out of the [Orengo lab](https://www.ucl.ac.uk/orengo-group/welcome-christine-orengos-group) at UCL.}</b>

### Data

- Genomic data
  - East London Genes and Health (ELGH) contains around 50,000 whole exomes of primarily Bangladeshi and Pakistani populations. Also contains electronic health records. https://www.genesandhealth.org/research/scientific-data-downloads
  - UK BioBank (UKBB) is a database that tracks the health of 500,000 participants, including genetic screens. Note that UKBB is not representative of general population. Skew towards population of Caucasian descent between the ages of 40-69.  https://www.ukbiobank.ac.uk/
  - Born in Bradford is a consortium tracking the health of about 13,000 participants from Bradford. Data includes EHR and genomes. https://borninbradford.github.io/datadict/bib/bib_biobank/current_samples/
  - deCODE is a research company that performed whole-genome sequencing experiments for about 2,500 individuals. Couldn't find open access to this dataset online. https://www.decode.com/
  - PROMIS (Patient-Reported Outcomes Measurement Information System) is a set of person-centred measures that evaluates and monitors physical, mental, and social health in adults and children. It can be used with the general population and with individuals living with chronic conditions. Also performed genotyping but could not immediately find this data. https://www.healthmeasures.net/explore-measurement-systems/promis 
  - ClinVar is a freely accessible, public archive of reports on the relationships among human variations and phenotypes, with supporting evidence. ClinVar thus facilitates access to and communication about the relationships asserted between human variation and observed health status, and the history of that interpretation. This resource can be used to assess e.g. haploinsufficiency. https://www.ncbi.nlm.nih.gov/clinvar/
  - OMIM is a comprehensive, authoritative compendium of human genes and genetic phenotypes that is freely available and updated daily. The full-text, referenced overviews in OMIM contain information on all known mendelian disorders and over 16,000 genes. OMIM focuses on the relationship between phenotype and genotype. https://www.omim.org/
  - The Genome Aggregation Database (gnomAD), originally launched in 2014 as the Exome Aggregation Consortium (ExAC), is a coalition of investigators seeking to aggregate and harmonize exome and genome sequencing data from a variety of large-scale sequencing projects, and to make summary data available for the wider scientific community. https://gnomad.broadinstitute.org/
  - ANNOVAR is an efficient software tool to utilize update-to-date information to functionally annotate genetic variants detected from diverse genomes (including human genome hg18, hg19, hg38, as well as mouse, worm, fly, yeast and many others). https://annovar.openbioinformatics.org/en/latest/ 

- Clinical trial status data for drug targets 
  - DrugBank aggregates clinical trial status data for particular targets in a structured fashion, e.g. https://go.drugbank.com/drugs/DB00001. Not immediately clear whether this data is also queryable in this structured way via API. 
  - ClinicalTrials.gov has an API that is queryable. However, organisation seems unstructured compared to DrugBank. https://clinicaltrials.gov/api/gui
  - DGIdb has a druggability assessment tool and drug-gene interaction search engine. The database is queryable through API. https://www.dgidb.org/api
  - Open Targets Genetics is a comprehensive tool highlighting variant-centric statistical evidence to allow both prioritisations of candidate causal variants at trait-associated loci and identification of potential drug targets. Accessible via GraphQL API. https://genetics-docs.opentargets.org/data-access/graphql-api
