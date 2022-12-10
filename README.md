# Credit card usage and default risk
[Slides](https://docs.google.com/presentation/d/1LXup8CaDlZFC_feaj2BzeHCIj5Nevm75pVJNJSUXTYo) |
[Executive Summary](https://docs.google.com/document/d/1OH3KMq_FwH_pfcZN56qx6OIA_jXwMwQan2YcQj9x-A0/) |
[Demo](https://huggingface.co/spaces/alex42t/CreditScore)
## Team members (Oak Team)
Alexander Timofeev

Martin Molina

## Motivation
Credit risk assessment of individuals is traditionally determined using data such as age, income, education level, and FICO or Equifax credit scores. The increasing popularity of e-commerce and the large amount of bank card transaction data that can be collected and associated with clients these days suggest that a machine learning approach that incorporates this data can be useful for credit risk determination. Using data about bank card transactions and default history of other clients, we built a model to assess the credit risk of an applicant that requires only his bank card transaction history. 

## Dataset
We used real data from Alfa Bank, released in the context of [this competition](https://ods.ai/competitions/dl-fintech-card-transactions), that contains 1.5 million records of approved credit applications for different products offered by the bank, and information on whether applicants eventually defaulted on them.

## Results and future directions
We were able to increase the proportion of applicants correctly predicted to default on a credit application in comparison with our baseline model, as measured by the ROC AUC metric.

Although our original objective was to rely solely on the anonymous data provided by the bank, our model could be improved if third party information and personal data were used such as credit scores and demographic information. However, as our model require no third party information about clients it could be used as a quick early screening tool to reduce application processing times and provide an easier experience for applicants.

## Requirements
The project requires a number of additional packages listed in `requirements.txt`. One option to install them is to run `pip install -r /path/to/requirements.txt`
