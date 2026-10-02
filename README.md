# 📊 Data Preprocessing
Data preprocessing is an important step in the Machine Learning workflow. It involves converting raw, incomplete, inconsistent, and unstructured data into a clean and suitable format for machine learning models.
Real-world datasets often contain missing values, duplicate records, incorrect data types, categorical values, outliers, and features with different scales. Data preprocessing helps solve these problems and improves the quality and reliability of the dataset.

---

## 🔄 Data Preprocessing Workflow
The typical data preprocessing workflow is:
Raw Data → Data Understanding → Data Cleaning → Data Transformation → Feature Engineering → Feature Selection → Data Splitting → Preprocessed Data

---

## 1. 📥 Data Collection
The first step is collecting data from different sources such as:
    * CSV files
    * Excel files
    * Databases
    * APIs
    * IoT sensors
    * Web applications
    * Public datasets
    * Real-time data streams
The collected data may contain numerical, categorical, textual, and date/time information.

---

2. 🔍 Data Understanding
Before preprocessing, the dataset needs to be understood properly.
Important things to examine include:
    * Number of rows and columns
    * Column names
    * Data types
    * Missing values
    * Duplicate records
    * Unique values
    * Numerical and categorical features
    * Minimum and maximum values
    * Mean, median, and standard deviation
    * Distribution of numerical features
This step helps identify problems that need to be handled during preprocessing.

---

## 3. 🧹 Data Cleaning
Data cleaning removes or corrects problems in the dataset.

### Missing Values
Missing values occur when information is not available for a particular record.
Common approaches include:
* Removing rows containing missing values
    * Removing columns with excessive missing values
    * Replacing missing numerical values with mean or median
    * Replacing missing categorical values with mode
    * Forward filling or backward filling for sequential data
The appropriate method depends on the dataset and the reason for missing data.

### Duplicate Data
Duplicate records are repeated observations in a dataset.
Duplicates can:
    * Increase dataset size unnecessarily
    * Affect statistical analysis
    * Bias machine learning models
Therefore, duplicate records should be identified and handled appropriately.

### Incorrect Data
Data may contain incorrect or inconsistent values, such as:
    * Negative age
    * Invalid dates
    * Incorrect category names
    * Impossible numerical values
    * Inconsistent spellings
These vales should be corrected or removed when appropriate.

---

## 4. 📐 Handling Data Types
Different columns may have different data types.
Common data types include:
    * Integer
    * Float
    * String
    * Boolean
    * Date/Time
    * Categorical data
Correct data types are important because machine learning algorithms generally require data in appropriate numerical formats.
For example, a column containing numbers stored as text should be converted into a numerical type before performing mathematical operations.

---

## 5. 🚨 Outlier Detection and Treatment
An outlier is an observation that is significantly different from the majority of the data.
For example, if most transaction amounts are between ₹100 and ₹10,000 but one transaction is ₹10,00,000, it may be an outlier.
Common methods for detecting outliers include:
    * IQR (Interquartile Range)
    * Z-Score
    * Box Plot
    * Statistical methods
    
Possible approaches to handling outliers include:
* Removing the outlier
* Capping extreme values
* Transforming the data
* Keeping the outlier when it represents a genuine observation

Outliers should not automatically be removed because they may contain important information.

---

## 6. 🔤 Encoding Categorical Data
Machine learning algorithms generally work with numerical values. Therefore, categorical data often needs to be converted into numerical representations.
Examples of categorical features:
    * Gender
    * City
    * Education
    * Transaction Type
    * Customer Occupation
Common encoding techniques include:

### Label Encoding
Assigns a numerical value to each category.
Example:
    **Low → 0**
    **Medium → 1**
    **High → 2**

### One-Hot Encoding
Creates separate binary columns for different categories.
For example, a City column containing Pune, Mumbai, and Delhi can be represented using separate indicator columns.

### Ordinal Encoding
Used when categories have a meaningful order.
Example:
  Low < Medium < High
The encoding technique should be selected according to the meaning of the categorical feature.

---

## 7. 📏 Feature Scaling
Different numerical features can have very different ranges.
For example:
    * Age: 18–80
    * Salary: 20,000–2,00,000
    * Transaction Amount: 100–1,00,000

    
Large numerical ranges can affect some machine learning algorithms.
Common scaling techniques include:
### Standardization:
Transforms features so that they generally have:
  * Mean ≈ 0
  * Standard deviation ≈ 1

### Normalization
Transforms values to a specific range, commonly:
**0 to 1**
Scaling is particularly important for algorithms based on distance or gradient optimization.

---

## 8. 🔄 Data Transformation
Data transformation changes the representation of data into a form that is more suitable for analysis or machine learning.
Examples include:
    * Log transformation
    * Power transformation
    * Numerical conversion
    * Date/time transformation
    * String transformation
    * Aggregation
    * Creating new features
For example, a date column can be transformed into:
    * Year
    * Month
    * Day
    * Day of week
    * Hour

---

## 9. 🛠️ Feature Engineering
Feature engineering involves creating useful new features from existing data.
For example, from:
**Transaction Date** and **Previous Transaction Date**

we can derive:
**Time Difference**
Similarly, from a date of birth we may derive:

**Age**
Good feature engineering can provide machine learning models with more useful information and improve predictive performance.

---

## 10. 🎯 Feature Selection
Feature selection means selecting the most relevant features for a machine learning problem.
Some datasets contain many unnecessary or redundant features.

Feature selection can help:
  * Reduce dataset complexity
  * Reduce training time
  * Remove irrelevant information
  * Reduce overfitting
  * Improve model interpretability
Features can be selected using statistical methods, correlation analysis, model-based techniques, or domain knowledge.

---

## 11. 📊 Handling Imbalanced Data
In classification problems, some classes may have significantly more observations than others.
For example:
    **Normal Transactions → 95%**
    **Fraudulent Transactions → 5%**
This is called class imbalance.

Common techniques include:
* Oversampling
* Undersampling
* Synthetic sampling
* Class weights
Handling imbalance is especially important in applications such as fraud detection, disease detection, and anomaly detection.

---

## 12. ✂️ Train-Test Split
After preprocessing and feature preparation, the dataset is commonly divided into separate subsets.
### Training Set
Used to train the machine learning model.

### Testing Set
Used to evaluate the model on unseen data.

A common division is:
**80% Training + 20% Testing**
Other divisions can also be used depending on the size and requirements of the dataset.

---

## 13. 🔬 Validation Data
For many machine learning projects, the dataset may be divided into three parts:
**Training Set → Validation Set → Test Set**
    * **Training:** Used to train the model
    * **Validation:** Used for model selection and hyperparameter tuning
    * **Testing:** Used for final evaluation
This helps provide a more reliable estimate of model performance.

---

## 14. ⚠️ Avoiding Data Leakage
Data leakage occurs when information from outside the training data is accidentally used during model training.
For example, calculating scaling parameters using the entire dataset before splitting it can allow information from the test set to influence the training process.

A safer workflow is:
**Split Data → Fit Preprocessing on Training Data → Transform Training Data → Transform Test Data**
This helps ensure that the test data remains unseen during model training.

---

## 15. 📈 Exploratory Data Analysis
Exploratory Data Analysis (EDA) is often performed before or alongside preprocessing to understand the dataset.
Visualization techniques include:
    * Histogram
    * Bar Chart
    * Line Chart
    * Scatter Plot
    * Box Plot
    * Pie Chart
    * Correlation Heatmap

EDA helps identify:
    * Data distributions
    * Relationships between variables
    * Trends
    * Outliers
    * Missing data patterns
    * Possible correlations

---

## 🧠 Why Data Preprocessing Is Important
Data preprocessing is important because raw data is rarely ready to be directly used by machine learning algorithms.
Effective preprocessing can:
    * Improve data quality
    * Reduce noise
    * Handle missing information
    * Remove inconsistencies
    * Convert categorical data into usable formats
    * Bring numerical features to comparable scales
    * Reduce the impact of outliers
    * Improve model training
    * Reduce computational complexity
    * Improve model performance
    * Reduce the possibility of overfitting

---

## 🏦 Example: Banking Fraud Detection
For a banking fraud detection system, raw transaction data may contain:
    * Transaction amount
    * Transaction date
    * Transaction type
    * Account information
    * Location
    * Device information
    * Login attempts
    * Account balance

The preprocessing workflow could involve:
                                **Raw Transactions**
                                ↓
                                **Handle Missing Values**
                                ↓
                                **Remove Duplicates**
                                ↓
                                **Correct Data Types**
                                ↓
                                **Handle Outliers**
                                ↓
                                **Encode Categorical Features**
                                ↓
                                **Create Time-Based Features**
                                ↓
                                **Scale Numerical Features**
                                ↓
                                **Select Important Features**
                                ↓
                                **Split Training and Testing Data**
                                ↓
                                **Machine Learning Model**

