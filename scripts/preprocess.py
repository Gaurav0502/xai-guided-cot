# modules used for data handling
import pandas as pd

# preprocessor for loan dataset
def preprocess_loan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the loan dataset by dropping irrelevant columns, handling NaNs,
    encoding categorical variables, and converting all columns to integers.

    Args:
        df (pd.DataFrame): Raw loan dataset.

    Returns:
        pd.DataFrame: Preprocessed loan dataset.
    """

    # drop irrelevant columns
    df = df.drop(columns=["loan_id"])

    # drop null values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    print(f"[LOAN] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    # encode categorical variables
    categorical_cols = [' education', ' self_employed']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.replace({True: 1, False: 0})
    df = df.replace({' Approved': 1, ' Rejected': 0})

    # convert all columns
    # to integer
    df = df.astype('int64')

    return df

# preprocessor for diabetes dataset
def preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the diabetes dataset by converting columns to numeric,
    handling NaNs, and returning the cleaned DataFrame.

    Args:
        df (pd.DataFrame): Raw diabetes dataset.

    Returns:
        pd.DataFrame: Preprocessed diabetes dataset.
    """

    # convert all columns
    # to numeric 
    df = df.apply(pd.to_numeric, errors="coerce")

    # drop null values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after
    print(f"[Diabetes] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    return df

# preprocessor for titanic dataset
def preprocess_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Titanic dataset by dropping irrelevant columns,
    encoding categorical variables, converting columns to numeric,
    handling NaNs, and returning the cleaned DataFrame.

    Args:
        df (pd.DataFrame): Raw Titanic dataset.

    Returns:
        pd.DataFrame: Preprocessed Titanic dataset.
    """

    # drop irrelevant columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # encode categorical
    # variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    embarked_map = {"S": 0, "C": 1, "Q": 2}
    df["Embarked"] = df["Embarked"].map(embarked_map)

    # convert all columns
    # to numeric 
    df = df.apply(pd.to_numeric, errors="coerce")

    # drop null values
    before = len(df)
    df = df.dropna()
    after = len(df)
    dropped = before - after

    print(f"[Titanic] Dropped {dropped} rows due to NaNs (kept {after} rows).")

    return df