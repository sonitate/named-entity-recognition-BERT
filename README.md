# Applying BERT/SciBERT token classifiers for named entity recognition (NER) on PGxCorpus

## Running an experience
For running an experience run the main code on an CPU or GPU device with the according parameters.
### Display parameters list :

`$ python main.py -h`

### Fine-tuning on PgxCorpus (cross validation) exemples:
* With BERT 
`$ python3 main.py -bert bert -F_type macro -lr 3e-5 -num_ep 10`

* With SciBERT
`$ python3 main.py -bert scibert -F_type macro -lr 3e-5 -num_ep 10`

* To save the model into a folder, add "--save"
e.g `$ python3 main.py -bert bert -F_type macro -lr 3e-5 -num_ep 10 --save`

The resulst of one experience should look like this :  
```
              precision    recall  f1-score   support

           0       0.89      0.90      0.90     23841
           1       0.74      0.71      0.72      2515
           2       0.56      0.17      0.26       113
           3       0.65      0.71      0.68      1465
           4       0.30      0.15      0.20       242
           5       0.79      0.84      0.82      4730
           6       0.83      0.79      0.81       850
           7       0.67      0.58      0.62      2470
           8       0.78      0.74      0.76      1532
           9       0.67      0.68      0.67      2767
          10       0.71      0.71      0.71       723

    accuracy                           0.82     41248
   macro avg       0.69      0.63      0.65     41248
weighted avg       0.82      0.82      0.82     41248
```
## Experiences statistics 

### Results files 
The experiences results are saved into three types of files :
* .res : 
  - Contain the precision, recall and fscore (micro or macro).
  - You can use this file for statistical performances computing.
* .pred : 
  - Contain model predictions end ground truth. 
  - You can use this file for error analysing.
* .loss_acc: 
  - Contain the the accuracy and loss function value with the training data.
  - You can use this file for observe models convergences.

### Statistics computing 
To compute results statistics run the stat code.

`$ python stat_result.py`

The results should looks like the following example : 
```
                min       max      mean       std
precision  0.659928  0.710321  0.686627  0.011053
recall     0.621679  0.649905  0.635854  0.007795
fscore     0.637523  0.663320  0.649737  0.007154
```

<img src="https://drive.google.com/uc?export=view&id=1606-ORWH1a4YAPgLyj_hrcvwknzuZGTp" width="500" height="300">
