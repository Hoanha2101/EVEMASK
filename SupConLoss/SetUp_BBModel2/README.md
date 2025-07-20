# ArcLoss - Supervised Contrastive Learning

## Folder map

```
TRAIN_SET/
├── class_0/
│   ├── 000000.jpg  
│   ├── ... 
├── class_1/
│   ├── 000000.jpg
│   ├── ...

TEST_SET/
├── class_0/
│   ├── 000000.jpg   ← used as reference image (key="000000")
│   ├── ... other test images ...
├── class_1/
│   ├── 000000.jpg
│   ├── ...
```

## How To Test
```
        +-----------------------------+
        |       TEST_SET Folder      |
        +-----------------------------+
                  |
                  |  For each class folder (e.g., class_0, class_1)
                  v
        +-----------------------------+
        |   Load reference image      |   <-- contains "key" (e.g., "000000")
        +-----------------------------+
                  |
                  v
        +-----------------------------+
        | Extract embedding vector    |
        |     using model             |
        +-----------------------------+
                  |
                  v
        +-----------------------------+
        |  Store label + embedding    |
        +-----------------------------+

Repeat above for all class folders
(Each class should contain 1 reference image)

============================

Now test phase begins:

        +------------------------------+
        | Loop over test images       |
        +------------------------------+
                  |
                  v
        +------------------------------+
        | Extract embedding from test |
        +------------------------------+
                  |
                  v
        +------------------------------+
        | Compare with all reference  |
        | embeddings (cosine similarity) |
        +------------------------------+
                  |
                  v
        +------------------------------+
        | Choose label with highest   |
        | cosine similarity           |
        +------------------------------+
                  |
                  v
        +------------------------------+
        | Save predicted label        |
        +------------------------------+

============================

Finally:

        +------------------------------+
        | Compare with true label     |
        | → Compute Accuracy          |
        | → Save fail/success cases   |
        +------------------------------+

```

## Kaggle Data

Test set (bbox): https://www.kaggle.com/datasets/khihon/logo-50

Train set (94 logo): https://www.kaggle.com/datasets/khihon/model2-94logo