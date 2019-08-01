# Objective
Develop a Skin Cancer Screener that can classify 7 diagnostic categories of skin lesions.
![main img](Figures/main_img.JPG)

# Success/Evaluation Criteria
if we pick at random (1 out of 7 diagnostic categories) then we will be correct 14.28% of the time. If we have a model that can predict the diagnostic categories at a rate of 50% or better, then that model will perform at least 4 times better then random chance and that would be a relatively successful model.

# Background
Skin cancer is the most common form of cancer in the United States with an annual cost of care exceeding $8 billion. With early detection, the 5 year survival rate of the most deadly form, melanoma, can be up to 99%; however, delayed diagnosis causes the survival rate to plummet to 23%.

[The HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) is a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10015 dermatoscopic images. This collection of images represent about 95% of the 7 diagnostic categories in the realm of pigmented lesions as seen in clinical settings: 

1. Actinic Keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
2. Basal Cell Carcinoma (bcc)
3. Benign Keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic Nevi (nv)
7. Vascular Lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

![AC](Figures/AC.png)
![BCC](Figures/BCC.png)
![BK](Figures/BK.png)
![DF](Figures/DF.png)
![Mel](Figures/Mel.png)
![Nevi](Figures/Nevi.png)
![Vasc](Figures/Vasc.png)

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id column within the HAM10000_metadatafile.

![Confusion matrix for CNN Classifier](Figures/CM_Best_Model.png)

# Process
The image categories were completely unbalanced. Almost 50% of images pertained to one category, another 20% were distributed between 2 other categories, and the final 30% were spread in the remaining 4 categories. We tried to run our model using various class weights to remedy this imbalance in data but this strategy yielded poor results.

Our dataset was broken down as follows: 64% training, 16% validation, 20% testing. We decided that we needd to augment our training data set. We created a function that would create various images from each of the under represented categories and manipulated like rotate, blurr, shear, etc. so we can balance the data set. Our training data set increased from over 6400 images to over 30000 images.


We built 3 Convolutional Neural Networks (CNN) and used transfer learning from 2 other pre-built models. Many iterations of each model and hyperparameter tuning were performed like:
- Adding/removing layers
- Batch Normalization
- Droupout
_ L2 Reguralization
- Learning Rate
- Dilation Rate
- Learning Rate on Plateau
- Optimizers: Stochastic Gradient Descent and ADAM
# 3. **Next Steps**
## 3a) **Model Improvement**
Our best model performed at a precision and recall of 76%. That model has 11 convolutional layers. We believe that in order to make it perform better we need to implement the following:

-  A lot more images for training
-  Modifying the model to include more convolution layers 
-  Experimenenting with hyperparameters such as: adaptable learning rates, Dropout, and L2 regularization
-  Develop an app for both IOS and Android: This will allow us to get users to submit the pictures of their skin lesions and will give our platform a continual stream of images to further train our model and refine it. Additionally, it will give the users a feedback whether they need to consult with a doctor or not.

## 3b) **Project Roadmap**
Obviously, beyond the training and model tweaking we need to have an infrastructure that can support images at scale and that is responsive. To that end, we will be using Amazon Web Services (AWS) to develop this architecture. When a user takes a picture in their app of their skin lesion, this image will be saved on an S3 bucket which will trigger a an AWS Lambda action. Lambda will invoke our convolutional neural network (CNN) which will be available as an end point on AWS Sagemaker. Once the model runs and processes the images, the result will be stored in another S3 bucket that will trigger another lambda event, this time a message back to the user in their app, detailing whether they should seek medical attention with a degree of confidence. Model predictions along with images file paths will and other relevant transactions will be saved in a Dynamo DB.

As we get more images and more training to our model, our degree of confidence will rise. Coupling the continuous training  with an evolving CNN and hyperparameter tuning, we believe that reaching a level of accuracy over 90% is certainly within reach. 

There is no reason why 2 people in the US should die from skin cancer every hour. We have a simple yet a powerful vision: 
**Snap a picture and potentially, save your life**.

![future_app](Figures/future_app.png)
