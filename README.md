# Model Deployment using AWS-Sagemaker-Docker-HuggingFace:  
   
### I. Reason for Model selection:               
        
I chose **DistilBERT** base uncased distilled **SQuAD** as my Huggingface model. DistilBERT is a transformers model, smaller and faster than BERT, which was pretrained on the same corpus in a self-supervised fashion, using the BERT base model as a teacher. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts using the BERT base model. More precisely, it was pretrained with three objectives:              
             
i. Distillation loss: the model was trained to return the same probabilities as the BERT base model.                            
  
ii. Masked language modeling (MLM): this is part of the original training loss of the BERT base model. When taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.         
           
iii. Cosine embedding loss: the model was also trained to generate hidden states as close as possible as the BERT base model.    
This way, the model learns the same inner representation of the English language than its teacher model, while being faster for inference or downstream tasks.
    
### II. Model_Deployment_Docker.ipynb: Post to Container endpoint & prints out the response:     
    
   i. Install Sagemaker SDK, Transformers in Python3   

  ii. Install necessary Libraries

  iii. Create SageMaker endpoint with the chosen model
  
   iv. Create Hugging Face Model Class

   v.  Deploy model to SageMaker Inference

   vi. Test with Sample Input

  vii. Run Predictor
  
 ### III. Dockerfile:

    i. Create Dockerfile  

   ii. Pull image from Dockerhub

  iii. Set nginx- server components to support multiple parallel incoming requests

   iv. Install necessary dependencies for SageMaker Inference Toolkit

   v.  Install pip requirements

   vi.  Install Hugging Face libraries and its dependencies

  vii.  Copy entrypoint script to the image

 viii. Copy the default custom service file to handle incoming data and inference requests
  
   ix. Build Docker Image 

   x. Define an entrypoint script for the docker image

   xi. Define command to be passed to the entrypoint

  xii. Run Inference and Check Docker Image in 'Docker Desktop'
     
 ## ML Models:

 ### 1. Naive Bayes Classifier:
 
![ALTTEXT](https://github.com/SaifurRR/ML-Model-Deployment-using-AWS-Sagemaker-Docker/blob/main/Na%C3%AFve%20Bayes%20Classifier.png)
 
 *Review :* [Bad : Good]
 
*Baby training set:* [[0.4980342 0.5019658]]

*Amazon Instant Video training set:* [[0.77906497 0.22093503]]

*Video Games training set:* [[0.32653528 0.67346472]]

### 2. Email Similarity Naive Bayes Classifier:

![ALTTEXT](https://github.com/SaifurRR/ML-Model-Deployment-using-AWS-Sagemaker-Docker/blob/main/Email_Similarity_NVM.jpg)

*Categories:* ['rec.sport.baseball' 'rec.sport.hockey']

*Corresponding labels:* [0 1]

*Naive Bayes Model Score:* 0.9723618090452262

### 3. Predict Baseball Strike Zones with SVM:    

<p align="center">
  <img src="https://github.com/SaifurRR/ML-Model-Deployment-using-AWS-Sagemaker-Docker/blob/main/3_baseball_strike_zones_ML.jpg" alt="Baseball Strike Zones" width="500"/>
</p>

In this project, we will use an SVM trained using a baseball dataset to find the decision boundary of the strike zone. The strike zone can be thought of as a decision boundary that determines whether or not a pitch is a strike or a ball. There is a strict definition of the strike zone — in practice, however, it will vary depending on the umpire or the player at bat.

<p align="center">
  <img src="https://raw.githubusercontent.com/SaifurRR/ML-Model-Deployment-using-AWS-Sagemaker-Docker/main/3_baseball_strike_zones_Aaron_Judge.jpg" alt="Baseball Strike Zones" width="500"/>
   <br>
  <b> </b> Baseball strike zones for Aaron Judge.
</p>

<p align="center">
  <img src="https://github.com/SaifurRR/ML-Model-Deployment-using-AWS-Sagemaker-Docker/blob/main/3_baseball_strike_%20zone_gamma_c_%20svmscore.jpg" alt="Baseball Strike Zones" width="700"/>
   <br>
    <b> </b> Optimize Hyperparameters </b>"gamma"</b> and </b>"C"</b>.
</p>

Best **SVM Score: 0.982** @ **gamma=160**, **C=140**.



