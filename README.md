# Ambiguous DGS
Deep Learning-based Classification of ambiguous signs in the German Sign Language

In this research project, 12 glosses of the German Sign Language (DGS) were selected and extracted from the Public DGS Corpus, creating a classification task with the glosses as classes. A gloss is a German word representing the meaning of a sign in DGS. Each gloss of the classes shares the same hand sign with another gloss, making that hand sign ambiguous and increasing the difficulty since a recognized hand sign could be classified as either gloss. That's why the model in this project doesn't only look at the hands, but also at the mouth to predict which gloss a video clip belongs to.

![Model Architecture](https://raw.githubusercontent.com/NPhamDinh/AmbiguousDGS/main/images/diagramm3.png)

| ROI  | Validation Accuracy | Test Accuracy |
| ------------- | ------------- | ------------- |
| Mouth  | 44.86%  | 40.67% |
| Upper Body  | 62.67%  | 63.33% |
| Mouth + Upper Body | **69.86%** | **68.00%** |
