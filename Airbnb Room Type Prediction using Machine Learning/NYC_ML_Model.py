'''Step1 :Import library '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as Sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import joblib
import warnings
warnings.filterwarnings("ignore")


'''Step2: Load Dataset in python'''
Border="-"*30
print(Border)
df=pd.read_csv("C:/Users/karti/Desktop/NYC_Airbnb_Open_Data/AB_NYC_2019.csv.zip")
print(Border)
print(df.head())
print(Border)
print(df.info())
print(Border)
print(df.describe(include='all'))
print(Border)
print(df.shape)

'''Step3: Exploratory Data Analysis (EDA)
We explore the data in the standard order every ML engineer follows:
Missing values
Univariate analysis (one variable at a time - distributions, skewness)
Bivariate analysis (relationship between features and the target)
Correlation between numeric features
Outlier inspection'''
#Missing values
missing=df.isnull().sum()
missing[missing>0]
print(missing)

#Target Variabal_Room Type
print(df['room_type'].unique())
print(Border)
print(df['room_type'].value_counts())

Sns.countplot(x='room_type',data=df)
#plt.show()

#UniVariate Analysis Numeric Features
Numeric_col=["price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count",
            "availability_365"]

df[Numeric_col].hist(bins=30,figsize=(12,8))
plt.show()

#UniVariate Analysis_Features VS Target
Sns.countplot(data=df,x='neighbourhood_group')
plt.show()

#Bivariate Analysis_Features Vs Target
Sns.boxenplot(x='room_type',y='price',data=df)
plt.show()

#Correlation Between Numeric features
corr=df[Numeric_col+["latitude","longitude"]].corr()
Sns.heatmap(corr,annot=True)
plt.show()

#Geographic Distribution (Bonus Visual)
Sns.scatterplot(x='latitude',y='longitude',data=df,hue='room_type')
plt.show()

'''Steps:4 Data Cleaning & Feature Engineering

1. Drop columns that are pure identifiers or free text and carry no generalizable
signal for a tabular model (id, name, host_id, host_name, last_review).

2. Fill missing reviews_per_month with e (no reviews yet).

3. Cap extreme outliers in price and minimum_nights using percentile clipping, 
so a handful of data-entry errors (e.g. $10,000/night, 1,250 minimum nights) don't distort the model.

4. Separate features (x) from the target (y).''' 
print(Border)
print("Data Cleaning & Feature Engineering")
print(Border)

#Drop columns That help us...
df_clean=df.drop(columns=['id','name','host_id','last_review'])
print(df_clean)

#Fill missing reviews_per_month with e (no reviews yet).
df_clean['reviews_per_month']=df_clean['reviews_per_month'].fillna(0)

#Cap extreme outliers
price_cap=df_clean['price'].quantile(0.99) 
nights_cap=df_clean['minimum_nights'].quantile(0.99)

df_clean['price'] = df['price'].clip(upper=price_cap)
df_clean['minimum_nights'] = df['minimum_nights'].clip(upper=nights_cap)

print(df_clean)

''' Step 5 Train/Test Split
We hold out 20% of the data as a test set that is never touched during model 
selection or tuning - it is only used once, at the very end, to report the final, 
honest performance. stratify-y keeps the same class proportions in both splits, which
matters beca is imbalanced.'''
print(Border)
print("Train/Test Split")
print(Border)
X=df_clean.drop(columns=['room_type'])
Y=df_clean['room_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.33,
    random_state=42,
    stratify=Y
)
print("Train/Test Split sucessfull")


'''Step:6. Preprocessing: Column Transformer + Pipeline

Production ML code should never manually transform train/test data with 
separate lines-it's amor-proneandfont we build a single, reusable ColumnTransformer:

Numeric features
median imputation + standard scaling
Categorical features most-frequent imputation + one-hot encoding

This transformer will be the first step of every model pipeline below,
so preprocessing is learned only on and consistently everywhere (no data leakage).'''


print(Border)
Numeric_col=["price","latitude","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count",
            "availability_365","longitude"]

Categorical_col=["neighbourhood_group","neighbourhood"]

Numeric_pipline=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),
                               
                                ('scale',StandardScaler())
                                
                                ])

Categorical_pipline=Pipeline(steps=[('impute',SimpleImputer(strategy="most_frequent")),
                
                                    ('encode',OneHotEncoder(handle_unknown='ignore'))
                                    ])


#preprocessing
preprocessing=ColumnTransformer(transformers=[(
    "Numeric",Numeric_pipline,Numeric_col),
     ("Categorical", Categorical_pipline,Categorical_col)
     ])
print(preprocessing)

'''step 7 Trying Multiple Algorithms
We now compare several models, each wrapped in its own pipeline with the same preprocessor.
We evaluate each with 3-fold stratified cross-validation on the training set 
(never touching the test set yet), so the comparison is fair and robust.
Models tried:

1. Logistic Regression - simple, interpretable linear baseline

2. Decision Tree - a single non-linear model, prone to overfitting

3. Random Forest - an ensemble of trees, usually a strong general-purpose model

4. Gradient Boosting - sequential ensemble, often the most accurate of the classics'''



#class_weight="Balanced" tells the model to pay more attention to the minority class
# # or rare class (Shared Room)
# #but the problem is GradintBossting doesnt support class weight so we left it as it is


models = {"Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),

        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),

        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),

        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

#Accuracy alone can be misleading on imbalanced classes, so we use macro f1

#both: plain accuracy and macro f1 (which treats every class equally

for name, model in models.items():

    pipe = Pipeline(steps=[
        ("Preprocessor", preprocessing),
        ("Classifier", model)
    ])

    accuracy = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=3,
        scoring="accuracy"
    )

    macro_f1 = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=3,
        scoring="f1_macro"
    )

    print(
        f"{name} -> Accuracy: {accuracy.mean():.3f}, "
        f"F1: {macro_f1.mean():.3f}"
    )
    
'''Step 8. Hyperparameter Tuning

We pick the best-performing model from the comparison above (whichever one that turns 
out to be) and search over its key hyperparameters using Randomized SearchCV 
(cheaper than an exhaustive grid search, while still exploring the space well).
We keep optimizing for macro-F1 since the classes are imbalanced.'''
'''Step 8. Hyperparameter Tuning'''

best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessing),
    ("classifier", RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

param_distribution = {
    "classifier__n_estimators": [100, 150, 200, 300],
    "classifier__max_depth": [8, 12, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10]
}

search = RandomizedSearchCV(
    estimator=best_pipeline,
    param_distributions=param_distribution,
    n_iter=10,
    cv=3,
    scoring="f1_macro",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best Parameters:", search.best_params_)
print("Best CV Macro-F1:", search.best_score_)


'''step:10. Final Evaluation on the Test Set

This is the and time we touch the held-out test set. It gives us an
honest estimate of how the final tuned model would perform.c completely
unseen listings.'''

best_pipeline=search.best_estimator_
Y_pred=best_pipeline.predict(X_test)

print(f"Accuracy_score:{accuracy_score(y_test,Y_pred)}")
print(f"f1_score:{f1_score(y_test,Y_pred,average='macro')}")

#confusion_matrix
Sns.heatmap(confusion_matrix(y_test,Y_pred),annot=True,fmt=".2f",
            xticklabels=best_pipeline.classes_,yticklabels=best_pipeline.classes_)
plt.show()

'''11. Saving the Final Model Artifact

In production, the entire pipeline (preprocessing and model together)
is serialized as a single artifact, so a new listing can be scored
with one call no manual preprocessing required at inference time.'''

joblib.dump(best_pipeline, "Model_pipeline.pkl")