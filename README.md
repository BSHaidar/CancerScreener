# Cancer Screener
# 1. **Business Understanding**

## 1a) **Problem Definition**

Skin cancer is the most common form of cancer in the United States, with an annual cost of care exceeding $8 billion. With early detection, the 5 year survival rate of the most deadly form, melanoma, can be up to 99%; however, delayed diagnosis causes the survival rate to plummet to 23%.

The  dataset consists of 10015 dermatoscopic images. These collection of images represent almost 95% of 7 diagnostic categories in the realm of pigmented lesions as seen in clinical settings: 
- Actinic Keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevi (nv)
- Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id column within the HAM10000_metadatafile.

## 1b) **Success/Evaluation Criteria**

if we pick at random (1 out of 7 diagnostic categories) then we will be correct 14.28% of the time. If we have a model that can predict the diagnostic category 50% or better, than that model will perform at least 4 times better than random chance and that would constitute a successful evaluation criteria.

# 2. **Data Understanding**

## 2a) **Public Data**
[The HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) is a large collection of multi-source dermatoscopic images of common pigmented skin lesions

## 2b) Title


# 3. **Next Steps**
## 3a) **Model Improvement**
Project "next steps" include potential ideas to improve the model through feature engineering, parameter tuning, etc.
Product Roadmap
## 3b) **Project Roadmap**
Project "next steps" include ideas for future product improvements that further address the original business problem.
