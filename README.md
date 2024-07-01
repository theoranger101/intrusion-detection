<h1> NSL-KDD Intrusion Detection </h1>

This project aims to analyze the NSL-KDD dataset using various classification algorithms to detect and classify network intrusions. The classifiers used are Random Forest, KNeighbors, SVM, and Gradient Boosting.

Ensure you have the following packages installed in your Python environment:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

*Load the data*
* The script loads the training and test datasets using pandas.

*Data Preprocessing*
* Check for null values and duplicates.
* Convert attack types into binary flags (normal vs. attack) and map attack categories to numerical values.
* Encode categorical features into numerical values.

*Exploratory Data Analysis (EDA)*
* Visualize data distributions and relationships between features.
* Analyze attack distributions by protocol type, service, and flag.

*Apply Data Mining Techniques*
* Split the dataset into training and validation sets.
* Test multiple classifiers (Random Forest, k-NN, SVM, Gradient Boosting) for both attack flag detection and attack type classification.
* Evaluate model performance using cross-validation and accuracy scores.

*To execute the script, simply run it in your Python environment.*
