# DeepFM for MOFs (DeepFM4MOF)
---
This package provides implementation of DeepFM for prediction of gas adsorption properties of MOFs

```shell
├── utils.py   
│   ├── create_hmof_dataset  # date processing, devide the data based on arguements
├── layer.py  
│   ├── FM_layer    # FM component
│   ├── Dense_layer # Deep component
├── model.py  
│   ├── DeepFM      # DeepFM model
├── train.py 
│   ├── main        # train and show the results
```
---
If used to impute missing values in the material-property matrix:
```shell
python train.py --coldstart=false
```
If used to predict target properties using cold-start experiments:

```shell
python train.py --coldstart=true --property=1 --targetfrac=0.9
```
(--property (int, default=1) is the predicted property, range of it is 1 to 28, --targetfrac (float, default=0.9) is the fraction of the test set of the target property, say it's 0.9, then 90% of the target property is separated from the dataset, it will be uesd to test the trained model,range of it is 0 to 1)

### If you used this code in your paper, please cite [Finding the optimal CO2 adsorption material: Prediction of multi-properties of metal-organic frameworks (MOFs) based on DeepFM](https://www.sciencedirect.com/science/article/abs/pii/S1383586622016665).Thank you.