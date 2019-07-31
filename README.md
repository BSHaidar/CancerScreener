# Cancer Screener
# 1. **Business** **Understanding**

## 1a) **Problem Definition**
The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_idcolumn within the HAM10000_metadatafile.

## 1b) ** Success/Evaluation Criteria**
Criteria for success and/or evaluation are laid out in the Business Understanding section of the README.

## **Documentation**
Project has a README file clearly documenting each step in the CRISP-DM workflow.

# 2. **Data Understanding**

## 2a) **Public Data**
README.md file contains a hyperlink to where the public data can be found, with an explanation of the data source.

## 2b) **Confidential Data**
README.md file explains that this project used confidential data and will not be made publicly available. In the event that confidential data is not available, fake or anonymized is included as a substitute.

## 2c) **Example Data**
If a public or scraped dataset is used, the repo contains the dataset, if it is sufficiently small. If the dataset is too large to include in the repo, a shard of data is included that could be used to replicate the data pipeline (potentially with reduced model performance) without downloading or scraping additional data.


# 3. **Next Steps**
## 3a) **Model Improvement**
Project "next steps" include potential ideas to improve the model through feature engineering, parameter tuning, etc.
Product Roadmap
## 3b) **Project Roadmap**
Project "next steps" include ideas for future product improvements that further address the original business problem.
