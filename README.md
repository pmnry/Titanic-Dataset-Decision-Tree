# Titanic Dataset From Kaggle
## Goal
This repositery is aimed at comparing multiple ML models performances on a Classification problem namely the prediction of survival of passengers on the Titanic
## Roadmap
### EDA and visualization
We first perform simple EDA, analyzing the joint distributions of variables in the dataset. Then we compute the correlations among the dataset.
### Features engineering
> FamilySize: using SibSp and Parch to compute the size of the family onboard of each passenger
> isAlone: is the passenger alone ?
> NameLength: Computing the length of the name of each passenger
> Title: grouping the titles, if a title is of the type "Count", "Baron" or else the title is transformed to "Rare"
### Models testing
The following models are tested:
> Decision Tree
> Random Forest
> SVM
> Gradient Boosting
### Ensemble
> Future method not implemented yet.
