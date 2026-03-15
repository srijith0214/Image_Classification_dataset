# Model Comparison Report

Model                  Accuracy  Precision     Recall         F1    Time(s)
---------------------------------------------------------------------------
CustomCNN                0.9934     0.9894     0.9934     0.9914     2145.2
MobileNetV2              0.9969     0.9968     0.9969     0.9968     3004.2 ← BEST
EfficientNetB0           0.2099     0.1182     0.2099     0.0998     1586.4


## Per-Class Reports

### CustomCNN
```
                                  precision    recall  f1-score   support

                     animal fish       0.98      1.00      0.99       520
                animal fish bass       0.00      0.00      0.00        13
   fish sea_food black_sea_sprat       1.00      1.00      1.00       298
   fish sea_food gilt_head_bream       1.00      0.99      0.99       305
   fish sea_food hourse_mackerel       1.00      0.99      1.00       286
        fish sea_food red_mullet       1.00      1.00      1.00       291
     fish sea_food red_sea_bream       1.00      1.00      1.00       273
          fish sea_food sea_bass       0.99      1.00      0.99       327
            fish sea_food shrimp       1.00      1.00      1.00       289
fish sea_food striped_red_mullet       0.99      1.00      0.99       293
             fish sea_food trout       1.00      1.00      1.00       292

                        accuracy                           0.99      3187
                       macro avg       0.90      0.91      0.91      3187
                    weighted avg       0.99      0.99      0.99      3187

```

### MobileNetV2
```
                                  precision    recall  f1-score   support

                     animal fish       1.00      0.99      0.99       520
                animal fish bass       0.91      0.77      0.83        13
   fish sea_food black_sea_sprat       0.99      1.00      0.99       298
   fish sea_food gilt_head_bream       1.00      1.00      1.00       305
   fish sea_food hourse_mackerel       1.00      1.00      1.00       286
        fish sea_food red_mullet       1.00      1.00      1.00       291
     fish sea_food red_sea_bream       1.00      1.00      1.00       273
          fish sea_food sea_bass       0.99      1.00      1.00       327
            fish sea_food shrimp       1.00      1.00      1.00       289
fish sea_food striped_red_mullet       0.99      1.00      1.00       293
             fish sea_food trout       1.00      1.00      1.00       292

                        accuracy                           1.00      3187
                       macro avg       0.99      0.98      0.98      3187
                    weighted avg       1.00      1.00      1.00      3187

```

### EfficientNetB0
```
                                  precision    recall  f1-score   support

                     animal fish       0.30      0.73      0.43       520
                animal fish bass       0.00      0.00      0.00        13
   fish sea_food black_sea_sprat       0.00      0.00      0.00       298
   fish sea_food gilt_head_bream       0.00      0.00      0.00       305
   fish sea_food hourse_mackerel       0.00      0.00      0.00       286
        fish sea_food red_mullet       0.20      0.03      0.06       291
     fish sea_food red_sea_bream       0.00      0.00      0.00       273
          fish sea_food sea_bass       0.00      0.00      0.00       327
            fish sea_food shrimp       0.15      0.96      0.26       289
fish sea_food striped_red_mullet       0.00      0.00      0.00       293
             fish sea_food trout       0.40      0.01      0.01       292

                        accuracy                           0.21      3187
                       macro avg       0.10      0.16      0.07      3187
                    weighted avg       0.12      0.21      0.10      3187

```
