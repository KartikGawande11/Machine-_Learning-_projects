import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# Prediction Function
# ============================================================

def predict_car_price(model, x_one_encoding, scalar_one_hot):

    Border = "-" * 30

    print(Border)
    print("Enter Car Details")
    print(Border)

    car_model = input("Enter car model: ")
    year = float(input("Enter car year: "))
    transmission = input("Enter transmission: ")
    mileage = float(input("Enter mileage: "))
    fuel_type = input("Enter fuel type: ")
    tax = float(input("Enter tax: "))
    mpg = float(input("Enter MPG: "))
    engine_size = float(input("Enter engine size: "))

    # ---------------------------------------------------------
    # Create DataFrame for new car
    # ---------------------------------------------------------

    new_car = pd.DataFrame({
        "model": [car_model],
        "year": [year],
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuel_type],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engine_size]
    })

    print(Border)
    print("Input Data:")
    print(new_car)

    # ---------------------------------------------------------
    # One-Hot Encoding
    # ---------------------------------------------------------

    new_car = pd.get_dummies(
        new_car,
        columns=["model", "transmission", "fuelType"],
        drop_first=True,
        dtype=int
    )

    # ---------------------------------------------------------
    # Make columns same as training data
    # ---------------------------------------------------------

    new_car = new_car.reindex(
        columns=x_one_encoding.columns,
        fill_value=0
    )

    # ---------------------------------------------------------
    # Numerical columns
    # ---------------------------------------------------------

    numeric_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # ---------------------------------------------------------
    # Scaling
    # ---------------------------------------------------------

    new_car[numeric_cols] = scalar_one_hot.transform(
        new_car[numeric_cols]
    )

    print(Border)
    print("Processed Input:")
    print(new_car)

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    prediction = model.predict(new_car)

    print(Border)
    print("Predicted Ford Car Price:", prediction[0])
    print(Border)


# ============================================================
# Main Function
# ============================================================

def main():

    Border = "-" * 30

    # ========================================================
    # Step 1: Load Dataset
    # ========================================================

    print(Border)
    print("Step 1: Load the dataset into the Python application")
    print(Border)

    df = pd.read_csv("ford.csv")

    print(df.head())

    print(Border)
    print("Shape:")
    print(df.shape)

    print(Border)
    print("Information:")
    df.info()

    print(Border)
    print("Statistical Summary:")
    print(df.describe())

    print(Border)
    print("Missing Values:")
    print(df.isnull().sum())

    # ========================================================
    # Step 2: EDA
    # ========================================================

    print(Border)
    print("Step 2: EDA")
    print(Border)

    # Price Distribution

    plt.figure(figsize=(8, 6))

    sns.histplot(
        df["price"],
        bins=50,
        kde=True
    )

    plt.title("Ford Car Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")

    plt.show()


    # Correlation Heatmap

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True
    )

    plt.title("Correlation Heatmap")

    plt.show()


    # Year vs Price

    plt.figure(figsize=(12, 6))

    sns.boxenplot(
        data=df,
        x="year",
        y="price"
    )

    plt.xticks(rotation=90)

    plt.title("Year vs Price")

    plt.show()


    # Mileage vs Price

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="mileage",
        y="price"
    )

    plt.title("Mileage vs Price")

    plt.show()


    # Engine Size vs Price

    plt.figure(figsize=(8, 6))

    sns.boxenplot(
        data=df,
        x="engineSize",
        y="price"
    )

    plt.title("Engine Size vs Price")

    plt.show()


    # Transmission vs Price

    plt.figure(figsize=(8, 6))

    sns.boxenplot(
        data=df,
        x="transmission",
        y="price"
    )

    plt.title("Transmission vs Price")

    plt.show()


    # Fuel Type vs Price

    plt.figure(figsize=(8, 6))

    sns.boxenplot(
        data=df,
        x="fuelType",
        y="price"
    )

    plt.title("Fuel Type vs Price")

    plt.show()


    # Model vs Price

    plt.figure(figsize=(14, 7))

    sns.boxenplot(
        data=df,
        x="model",
        y="price"
    )

    plt.xticks(rotation=90)

    plt.title("Ford Model vs Price")

    plt.show()


    # ========================================================
    # Step 3: X and Y
    # ========================================================

    print(Border)
    print("Step 3: Prediction Input and Output")
    print(Border)

    X = df.drop(columns=["price"])

    Y = df["price"]

    print("X:")
    print(X)

    print("Y:")
    print(Y)


    # ========================================================
    # Step 4: One-Hot Encoding
    # ========================================================

    print(Border)
    print("Step 4: One-Hot Encoding")
    print(Border)

    x_one_encoding = pd.get_dummies(
        X,
        columns=[
            "model",
            "transmission",
            "fuelType"
        ],
        drop_first=True,
        dtype=int
    )

    print("One-Hot Encoded Data:")
    print(x_one_encoding)

    print("Shape:")
    print(x_one_encoding.shape)


    # ========================================================
    # Step 5: Label Encoding
    # ========================================================

    print(Border)
    print("Step 5: Label Encoding")
    print(Border)

    Xlabel = X.copy()

    categorical_cols = [
        "model",
        "transmission",
        "fuelType"
    ]

    for col in categorical_cols:

        encoder = LabelEncoder()

        Xlabel[col] = encoder.fit_transform(
            Xlabel[col]
        )

    print("Label Encoded Data:")
    print(Xlabel)

    print(Border)
    print("Model Value Counts:")

    print(
        Xlabel["model"].value_counts()
    )


    # ========================================================
    # Step 6: Scaling One-Hot Data
    # ========================================================

    print(Border)
    print("Step 6: Scaling One-Hot Encoded Data")
    print(Border)

    numeric_cols = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]

    # IMPORTANT:
    # Separate scaler for One-Hot data

    scalar_one_hot = StandardScaler()

    x_one_encoding[numeric_cols] = (
        scalar_one_hot.fit_transform(
            x_one_encoding[numeric_cols]
        )
    )

    print("Scaled One-Hot Data:")
    print(x_one_encoding)

    print("Shape:")
    print(x_one_encoding.shape)


    # ========================================================
    # Step 7: Scaling Label Data
    # ========================================================

    print(Border)
    print("Step 7: Scaling Label Encoded Data")
    print(Border)

    # Separate scaler for Label data

    scalar_label = StandardScaler()

    Xlabel[numeric_cols] = (
        scalar_label.fit_transform(
            Xlabel[numeric_cols]
        )
    )

    print("Scaled Label Data:")
    print(Xlabel)

    print("Shape:")
    print(Xlabel.shape)


    # ========================================================
    # Step 8: Train/Test Split - One-Hot
    # ========================================================

    print(Border)
    print("Step 8: One-Hot Model Train/Test Split")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        x_one_encoding,
        Y,
        test_size=0.20,
        random_state=42
    )

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)


    # ========================================================
    # Step 9: Linear Regression
    # ========================================================

    print(Border)
    print("Step 9: Linear Regression")
    print(Border)

    model = LinearRegression()

    model.fit(
        X_train,
        Y_train
    )

    print("Model trained successfully")


    # ========================================================
    # Step 10: Prediction on Test Data
    # ========================================================

    Y_pred = model.predict(
        X_test
    )

    print(Border)
    print("Y Prediction:")
    print(Y_pred)

    print(Border)
    print("Y Test:")
    print(Y_test)


    # ========================================================
    # Step 11: Evaluation
    # ========================================================

    print(Border)
    print("Step 11: Model Evaluation")
    print(Border)

    mae = mean_absolute_error(
        Y_test,
        Y_pred
    )

    mse = mean_squared_error(
        Y_test,
        Y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        Y_test,
        Y_pred
    )

    print("MAE  :", mae)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("R2   :", r2)


    # ========================================================
    # Step 12: Actual vs Predicted Graph
    # ========================================================

    print(Border)
    print("Step 12: Actual vs Predicted")
    print(Border)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        Y_test,
        Y_pred
    )

    plt.xlabel("Actual Price")

    plt.ylabel("Predicted Price")

    plt.title(
        "Actual vs Predicted Ford Car Prices"
    )

    plt.plot(
        [Y_test.min(), Y_test.max()],
        [Y_test.min(), Y_test.max()]
    )

    plt.show()
    # ========================================================
    # Step 13: User Input Prediction
    # ========================================================

    print(Border)
    print("Step 13: Predict Price for New Car")
    print(Border)

    predict_car_price(
        model,
        x_one_encoding,
        scalar_one_hot
    )
# ============================================================
# Program Starting Point
# ============================================================

if __name__ == "__main__":

    main()