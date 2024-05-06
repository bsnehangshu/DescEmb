# CS 598 Deep Learning for HealthCare - Final Project
***Team 41***

Indranil Guha| Iguha4@illinois.edu  
Snehangshu Bhattacharjee | sb8@illinois.edu


# Introduction
The paper can be found in this link: https://arxiv.org/abs/2108.03625


**Citation**

      @misc{
      hur2021unifying,
      title={Unifying Heterogenous Electronic Health Records Systems via Text-Based Code Embedding},
      author={Kyunghoon Hur and Jiyoung Lee and Jungwoo Oh and Wesley Price and Young-Hak Kim and Edward Choi},
      year={2021},
      eprint={2108.03625},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      }

***Background of the problem***

Increased adoption of electronic health record (EHR) systems offers great potential for EHR-based predictive models to improve healthcare quality. But EHR data is heterogenous in nature and contemporary EHR data rely on data systems ranging from standardized codes e.g ICD9, LOINC etc. to free text entry. The commonly used “code based embedding” or CodeEmb approach used in different hospitals thus are not transferrable, and as a result predictive models couldn’t be trained on large EHR data collected from various hospitals or medical institutions due to its heterogenous format, so these predictive models can be used at scale in order to get its full effectiveness. This challenge can be addressed using common data model such as OMOP, FHIR (Rajko- mar et al., 2018), however this requires significant human effort and domain knowledge and different code systems used across the hospitals also may not always be mapped to a common model.

***Paper explanation***

In this paper, author suggested a code-agnostic text based representation learning called Description-Based Embedding or DescEmb. DescEmb adopts a Neural Text Encoder to convert the medical codes to a contextualized embedding which allows codes from different systems maps to the same text embedding space. As compare to code-based embedding, In this specific approach, instead of directly embedding the medical codes, a series of vector representation of code descriptions passed through the neural text encoder and this DescEmb makes it possible to train models on differently formatted EHR data interchangeably due to its code-agnostic nature.

The Text Encoder in DescEmb model can be any model that can generate a hidden state representation from a given description. The paper used two model architecture as a text encoder : Bi-Directional RNN and BERT. It also mentioned about pretraining the text-encoder using Masked Language Modeling (MLM). In the context of drug prescriptions, dosage or rate of Infusion can be useful to know the patient status. Value embedding is another innovative approach which is introduced here to capture the values along with the description added another effective feature for effective predictive outcome. Four different value embedding methods has been used such as – Value Aggregation (VA), Digit split value aggregation (DSVA), Digit Place embedding (DPE), and Value concatenated embedding (VC).

The code agnostic approach described in the paper proposes improvement in performance with Zero Shot and few shot transfer learning and pooled learning and thus it unifies heterogeneous code systems in healthcare predictive research.

Contributions of this specific approach to the predictive healthcare research:

1)	DescEmb achieves comparable or superior performance to CodeEmb on common clinical predictive tasks.

2)	Two differently structured EHR can be used to train and test predictive models interchangeably showing better performance than training on a single EHR data.

3)	No additional domain knowledge or manual effort needed to pool two differently structured EHR dataset into the embedding space.

4)	It opens the door for text based approach in predictive healthcare research.

#Scope of Reproducibility

In this project we will be experimenting on the below two hypothesis as a scope of reproducibility.

***Hypothesis 1***

We will Pretrain the embedding model "descemb_rnn" with MLM on MIMIC III database using bidirectional RNN architecture as porposed in the paper and will observe how the model is trained and will evaluate based on average loss by varying no of epoch and learning rate.

Then we will Pretrain the embedding model "codeemb" with W2V on same MIMIC III database and will observe how the model is trained and will observe if descemb_rnn shows improved average loss in comparison with the codeemb.

***Hypothesis 2***

We will preprocess MIMIC III data using the code present in the github repo mentioned in the paper for a predictive task "mortality" and using this preprocessed data we will train and validate the ehr_model first using descemb_rnn as the embedding model and RNN at the prediction layer with value embedding method "VA" and observe the average loss and precision of model, next we will be using the same preporcessed data to train and validate the ehr_model using "codemb" as the embedding model and RNN at the prediction layer and observe if the ehr_model with decemb as embedding model shows better perfromance. This will establish our experiment in single domain learning setting.

***Ablation***

We can see how value embedding is impacting the performance of a predictive task using a DescEmb model by training w/ and w/o value embedding method. Here we will train and validate the ehr_model using the same pre processed data as mentioned in hypothesis #2, decemb_rnn as embedding model, RNN at the prediction layer and ommit the value embedding method to see if the model performance with value embedding method as mentioned in hypothesis #2 is still better than the model w/o value embedding method.

***Note***
 Due to limitation of computing resource required to complete preprocessing and model traning we could not run this experiment with full MIMIC III database, instead we used the MIMIC III demo database for this experiment. Due to less number of sample data for training, we observed the model is overfitting which is undertstandable but we could still make some meaningful observation about the validatity in the claim made by the authors of the paper.

## Environment Setup
All required packages and libraries are present in the python 3 default colab runtime excpet below , run the code cell to install this. Refer to envioronment.yml file for more details about the requirement.

# Check and Install dependencies
!pip install iterative-stratification

## Source Code Setup

Existing code in the GitHub repo referenced in the paper will be used to run experiments on proposed hypothesis. https://github.com/hoon9405/DescEmb

Steps to setup your google drive:

1. Create a folder "Project" under Mydrive

1. Clone from the github https://github.com/guhaindranil83/DescEmb.git to make the local repo "descemb".

2. Upload the local repo "descemb" to "Project" folder.

4. Download the data and setup the directory for Input path as prescribed in the Data section below.

Mount google drive to the colab runtime.
from google.colab import drive
drive.mount('/content/drive')
## Data


### DATA Download Instruction

***Download the datasets from the MIMIC III Demo database for this experiment, the demo database is used to run this experirment as part of this project submission. Only MIMIC III Dataset is used for the scope of this project***

Johnson, A., Pollard, T., & Mark, R. (2019). MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. https://doi.org/10.13026/C2HM2Q


***Note:***
Headers in all csv file must be in CAPS for the preprocessing code to work, otherwise you will get key error when processing the input datasets.
***MIMIC III and eICU - Data Source used in the Paper***

Full Datasets can be downloaded from MIMIC III and eICU, filenames and data statistics are given in the below tables.

**MIMIC III**

Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26.

**eICU**

Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). eICU Collaborative Research Database (version 2.0). PhysioNet. https://doi.org/10.13026/C2WM1R


    data_input_path
    ├─ mimic
    │  ├─ ADMISSIONS.csv
    │  ├─ PATIENTS.csv
    │  ├─ ICUSYAYS.csv
    │  ├─ LABEVENTES.csv
    │  ├─ PRESCRIPTIONS.csv
    │  ├─ PROCEDURES.csv
    │  ├─ INPUTEVENTS_CV.csv
    │  ├─ INPUTEVENTS_MV.csv
    │  ├─ D_ITEMDS.csv
    │  ├─ D_ICD_PROCEDURES.csv
    │  └─ D_LABITEMBS.csv
    ├─ eicu
    │  ├─ diagnosis.csv
    │  ├─ infusionDrug.csv
    │  ├─ lab.csv
    │  ├─ medication.csv
    │  └─ patient.csv
    ├─ ccs_multi_dx_tool_2015.csv
    └─ icd10cmtoicd9gem.csv

    ```
    ```
    data_output_path
    ├─mimic
    ├─eicu
    ├─pooled
    ├─label
    └─fold
    ```



## Hypothesis#1 : Pretraining from Scratch


### Data Preprocessing Code and Command

# First change the directory to preprocess folder
%cd /content/drive/MyDrive/Project/DescEmb/preprocess/
# Install necessary utility files
!python preprocess_utils.py
## use --data_type pretrain for Hypothesis #1
!python preprocess_main.py --src_data mimiciii --dataset_path /content/drive/MyDrive/Project/DescEmb/data_input_path/mimic --dest_path /content/drive/MyDrive/Project/DescEmb/data_output_path --data_type pretrain --target_task mortality


### Model
Author of this paper used two model architecture for text encoder Bi-RNN and BERT. Here we will pretrain the text encoder using Bi-RNN architecture on MIMIC III Database using MLM for the DescEmb model. Value encoding is set to 'NV' as it does not applicable for pretraining. For CodeEmb text encoder, standrad Word2Vec is used.


### Training

***Hyperparams***

* lr is set to 0.001, we tried to run the pretraining with smaller lr 0.0001 but the pretraining time gets increased significanlty and we didn't observe any meaningful impromevement, so we kept the lr as .001 for this script.

* n_epochs set to 50, at this epoch model is still not converging but we kept it at 50 for reasonable runtime. Results with n_epoch = 500 is shown below for reference.

* Other model parameters are set to default, refer to
main.py for more details about deafult params setting.

   * --ratio : 100 (used 100% of data for the training due to limited availibility in demo database)
   * --batch_size : 128
   * --enc_embed_dim : 128
   * --enc_hidden_dim : 256
   * --mlm_prob : 0.3


%cd /content/drive/MyDrive/Project/DescEmb/
**Pre-train a DescEmb model with Masked Language Modeling (MLM)**
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path --model descemb_rnn --src_data mimiciii --ratio 100 --n_epochs 50 --lr .001 --value_mode NV --task mlm
# input_path must be preproces destination path
# n_epochs 50
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path --model descemb_rnn --src_data mimiciii --ratio 100 --n_epochs 500 --lr .001 --value_mode NV --task mlm
# input_path must be preproces destination path
# n_epochs 500
**Pre-train a CodeEmb model with Word2Vec**
#!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path/mlm --model codeemb --src_data mimiciii --ratio 100 --n_epochs 50 --task w2v
# input_path must be preproces destination path
# n_epochs 50
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path/mlm --model codeemb --src_data mimiciii --ratio 100 --n_epochs 500 --task w2v
# input_path must be preproces destination path
# n_epochs 500

### Evaluation
Cross entropy loss is calculated for pretraining the models for both MLM and W2V task setting.
Average loss will be compared to see the pretraining performance of CodeEmb vs DescEmb to validate the hypothesis.


## Hypothesis#2/Ablation : Single Domain Learning

### Data Preprocessing Code and Command
Preprocessing needs to be done for predictive learning without data_type "Pretraining" argument.

***Preprocessing code***

Refer to DescEmb/Preprocess/preprocess_main.py
%cd /content/drive/MyDrive/Project/DescEmb/preprocess/
## --data_type pretrain removed for pre proessing data for predictive tasks
!python preprocess_main.py --src_data mimiciii --dataset_path /content/drive/MyDrive/Project/DescEmb/data_input_path/mimic --dest_path /content/drive/MyDrive/Project/DescEmb/data_output_path

### Model
As proposed in the paper, ehr_model has two layer - embedding layer which can produce the hidden representation from a given description using neural text encoder and then at the prediction layer RNN model architecture is used to predict many to many or many to one prediction from the output of embedding layer.

In single domain learning setting, we used MIMIC III demo database for training and validation of the model first with DescEmb as the embedding model and then CodeEmb as embedding model. For evaluation,  we compared the average training loss, average precision score to observe which model perform better.

DescEmb embedding model is built as Bi-RNN textencoder and CodeEmb embedding model is built using standrad Word2Vec model architecture.

This experient is run for a predictive task "mortality".

Also, we used Value embedding mode "VA" and "NV" for DescEmb embedding model training to see how the value embedding impacts the performance as mentioned in the "**Ablation**" scope of this document.


### Training

***Hyperparameter settings***

We tried to observe results by changing learning rate and n_epochs.

* Due to challange in computation resource we have produced the results with lr .001

* n_epochs = 10. Due to smaller size of demo data, the model stopped at lower epochs as the precision didn't change much with multiple epochs. however, we could observe descemb performs better than codeemb.

* Other model parameters are set to default, refer to
main.py for more details about deafult params setting.

   * --batch_size : 128
   * --enc_embed_dim : 128
   * --enc_hidden_dim : 256
   * --rnn_layer : 1
   * --dropout : 0.3
   * --pred_embed_dim : 128
   * --pred_hidden_dim : 256
   * --max_event_len : 150
   * --task : mortality

%cd /content/drive/MyDrive/Project/DescEmb/
***Train the ehr_model with DescEmb_rnn as embedding model, RNN at the prediction layer, and task "mortality" and value_mode "VA"***
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path --model ehr_model --embed_model descemb_rnn --pred_model rnn --src_data mimiciii --ratio 100 --n_epochs 10 --value_mode VA --task mortality
# input_path must be preproces destination path
# Here VA (Value Aggregation) is selected as value embedding
***Train the ehr_model with CodeEmb as embedding model, RNN at the prediction layer, and task "mortality"***
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path --model ehr_model --embed_model codeemb --pred_model rnn --src_data mimiciii --n_epochs 10 --ratio 100 --task mortality
# input_path must be preproces destination path
***Train the ehr_model with DescEmb_rnn as embedding model, RNN at the prediction layer, and task "mortality" and value_mode "NV"***
!python main.py --distributed_world_size 1 --input_path /content/drive/MyDrive/Project/DescEmb/data_output_path --model ehr_model --embed_model descemb_rnn --pred_model rnn --src_data mimiciii --ratio 100 --n_epochs 10 --value_mode NV --task mortality
# input_path must be preproces destination path
# Here NV (No Value) is selected as value embedding - for Ablation
### Evaluation
average Loss and average precision score are compared to see the performance of ehrmodel with CodeEmb and DescEmb as the embedding model to validate this hypothesis.

***For ablations,***
average Loss and average precision score will be compared to see the impact of value embedding for ehrmodel using DescEmb as the embedding model in comparison to the same model without value embedding.