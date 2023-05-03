# Ambiguous DGS
Deep Learning-based Classification of ambiguous signs in the German Sign Language

In this research project, 12 glosses of the German Sign Language (DGS) were selected and extracted from the Public DGS Corpus, creating a classification task with the glosses as classes. A gloss is a German word representing the meaning of a sign in DGS. Each gloss of the classes shares the same hand sign with another gloss, making that hand sign ambiguous and increasing the diffculty since a recognized hand sign could be classified as etiher gloss. That's why the model in this project doesn't only look at the hands, but also at the mouth to predict which gloss a video clip belongs to.

[!Model Architecture](https://github.com/NPhamDinh/AmbiguousDGS/blob/main/images/diagramm3.png)
