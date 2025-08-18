# AOXP
antioxidant peptide predict
Code and trained model for our paper **Antioxidant Peptide Prediction Based on Multi-Scale Characterization: A Method Combining Protein Large Language Models and Physicochemical Features**
![Overall FrameWork](https://github.com/zhangzhangleilei/AOXP/blob/main/fig1.tif)

<br>

## Installation
AOXP can be downloaded by following the commands below.
```bash
git clone https://github.com/zhangzhangleilei/AOXP.git
cd AOXP
conda env create -f environment.yml -n AOXP
conda activate AOXP
```
<br>

## Data
The dataset used can be downloaded from [data](https://github.com/zhangzhangleilei/AOXP/tree/main/data)
<br>

## Predict
We have provided AOXP model for you to use [predict](https://github.com/zhangzhangleilei/AOXP/tree/main/predict)
If you want to use our model, please first generate the features of your own dataset according to the following code.
```bash
cd embedding

python esm_embedding.py [path]

python GP_CTD_embedding.py [input_path] [save_path] 
```
The above command will generate fea1.csv and fea2.csv. Execute the following command to complete the prediction.
```bash
cd predict

python predict.py [fea1] [fea2] [path] [k] #k, feature selection index [2400]
```
<br>

## Web
We have developed a web server for the above process to facilitate its usage.
```bash
http://aidd.bioai-global.com/anoxp/
or
http://218.244.151.86/anoxp/
```
<br>

## Contact
If you have any problems with downloading or using the model, please contact zhangleilei0327@163.com. We will reply in a timely manner upon seeing your message.
