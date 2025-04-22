## Background
Our group participated in a Machine Learning Kaggle competition for CS 363M: Principles of Machine Learning I. We were tasked to predict the outcomes of animals entering the Austin Animal Center, and after 3 weeks, our team was ranked **#8 out of the 108 teams** that participated (top 7%). Our model was evaluated on balanced accuracy and you can find the models we used and the resulting accuracies in the Final Report as well as other supplementary reports. 

### Important Notes

Our group kept a collection of notes throughout the competition about what we did and when we did it. 

- Any CSV file with the appending “test_” in the name indicates a prediction we submitted.
- There are some files with the name “_normalized” - please ignore these, as these are COPIES of the original files. These copies were generated in order to call other functions across different python notebooks. Sometimes, python notebook files are missing necessary metadata to be ran in other notebooks. The normalized version fixes this by adding the necessary metadata.
- All files have attempted cross validation via the cross_val_score methods, with necessary preprocessing for models internally.

---

### File Notes

**balanaced_random.ipynb**
> This was an attempt to use a Balanced Random Forest ensemble classifier that accounted for class imbalances. This used stratified k-folding and a balanced accuracy score. The name is misspelled.

**breed_mapping.log**
> A log created by clustered_dog_breeds that tracked which breed was mapped to which cluster. Created for visibility.

**catboost.ipynb** 
> This is a version of catboost where we tried to implement higher thresholding for predictions on the majority class. This ended up creating a bug that we couldn’t figure out, and so we switched to catboost_working.ipynb. 

**catboost_working.ipynb**
> Standard catboost using stratified k-folding and class weighting. 

**clustered_dog_breeds.ipynb**
> k-means clustering of dog breeds using the dogs_cleaned.csv and the breed column of the train dataset. We determined best k ~ 9 using both a silhouette score plot and the elbow method. 

**dog_info.csv**
> A version of dogs_cleaned.csv created by clustered_dog_breeds.ipynb that kept features we felt were relevant for outcome potential. 

**dog_cleaned.csv**
> CSV file of traits of different dog breeds obtained from a Kaggle set online: https://www.kaggle.com/datasets/yonkotoshiro/dogs-breeds?select=dogs_cleaned.csv. Used to see if we could get additional insights for dogs in dataset	

**exploration.ipynb**
> This is a collection of how we explored different features about the train data given in order to figure out what direction to go through

**lightgbm.ipynb**
> This is a converted version of xgboost that instead uses lightgbm, another gradient boosting tree model. Near the deadline, we experimented with dropping more columns and adding additional features to try and squeeze out any extra performance out of a different boosting model.

**log_reg.ipynb**
> This was an attempt to use a Logistic Regression model that uses linear modeling for classification. This used stratified k-folding, cross validation, and a scoring with class weights accounts for. Previously attempts of SMOTENC were in the file before we stopped utilizing it. 

**ml_project.ipynb**
> This is a collection of functions used across all files as well as the data cleaning process, which includes standardizing age in months, creating columns for intake month and intake hour, sorting the colors into similar groups, dropping unnecessary columns, and clustered breeds. Additional functions including StratifiedKFold generation, creating a new scoring metric for balanced_accuracy

**multi_ovr_random_forest.ipynb**
> Trained multiple one vs. rest models using random forest as the base classifier. Abandoned after the accuracies came out low. 

**neural_net.ipynb**
> Standard neural net classifier. Only ran once after we decided to try other methods that suited categorical features better. 

**naive_bayes.ipynb**
> This has methods for both a MultinominalNB model and a CategoricalNB model. Multinomial gave some success initially when using with SMOTENC until we realized that there was data leakage. After refactoring, neither CategoricalNB nor MultinomialNB was given good accuracies, due to defying the naive assumption, since some of the model features given were NOT independent from each other (ex: breed and color).

**ovr_xg_boost.ipynb**
> Trained multiple one vs. rest classifiers using XGBoost as the base classifier with threshold voting. Standardized prediction percentages using calibration. Used stratified k-folding, threshold optimization through a randomized search, majority class downsampling, and heat mapping for visualization. 

**random_forest.ipynb**
> Random forest classifier using stratified k-folding and class weighting.

**svm.ipyb**
> Attempt to use a LinearSVC to see if SVC model could do predictions. 

**voting.ipynb**
> This was an attempt to use a heterogeneous ensembling method via voting. VotingClassifier did not work as expected, so we made a manual voting code with thresholding. It calls several other classifiers created in github, hence the “_normalized” files in the system. 

**xg_boost.ipynb**
> xgboost was the first tree ensemble we used after attempting to use decision trees for our model. It ended up having the highest accuracy in the end, largely because of our extensive feature engineering and how effective xgboost is at differentiating class labels. Most ideas we had were tried in this file, such as frequency encoding, SMOTE, one hot encoding, StratifiedKFolds, class weighting, and balanced accuracy.

**xg_cat_logistic_stack.ipynb**
> Here, we attempted to stack together xgboost and catboost, our two best models. This ended up not succeeding because the model took a significant amount of time to run. The demands of training multiple models also stressed out our machines’ memory, which added to the runtime of training. In the end, we were not able to get a successful set of predictions from this stacking classifier.

**x_random_trees.ipynb**
> An extra random trees model. Uses stratified k-folding and class weighting. We eventually abandoned this in favor of random forests due to balanced accuracy performance. 
