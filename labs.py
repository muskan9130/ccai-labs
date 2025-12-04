# ---------------- SINGLE NOTEBOOK CELL ----------------
import json, io, boto3, sagemaker, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# ====== Local CSV path for house price dataset ======
YOUR_LOCAL_PATH = "/path/to/your/house_price.csv"   # <-- CHANGE THIS

def load_local_csv(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from local file: {path}")
    return df


# ====== IAM CREATION (ML / DATA / DEVOPS USERS) ======
iam = boto3.client("iam")
def create_user_minimal(name):
    try:
        iam.create_user(UserName=name)
    except Exception as e:
        print("Exists / Error:", e)
    return f"iam://{name}"


# ====== IAM POLICY: Only Amazon Comprehend ======
COMPREHEND_ONLY_POLICY = {
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow","Action":["comprehend:*"],"Resource":"*"}]
}

def create_comprehend_policy(name="ComprehendOnlyPolicy"):
    resp = iam.create_policy(PolicyName=name, PolicyDocument=json.dumps(COMPREHEND_ONLY_POLICY))
    return resp["Policy"]["Arn"]


# ====== SageMaker Role restricted to one S3 bucket ======
def create_sagemaker_role_for_bucket(role_name, bucket_name):
    assume = {
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow",
                    "Principal":{"Service":"sagemaker.amazonaws.com"},
                    "Action":"sts:AssumeRole"}]
    }

    try:
        r = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    except:
        r = iam.get_role(RoleName=role_name)

    role_arn = r["Role"]["Arn"]

    s3_policy = {
      "Version":"2012-10-17",
      "Statement":[
        {"Effect":"Allow",
         "Action":["s3:GetObject","s3:PutObject","s3:ListBucket"],
         "Resource":[f"arn:aws:s3:::{bucket_name}",
                     f"arn:aws:s3:::{bucket_name}/*"]}
      ]
    }

    iam.put_role_policy(RoleName=role_name,
                        PolicyName=f"{role_name}-S3Access",
                        PolicyDocument=json.dumps(s3_policy))

    return role_arn


# ====== Block S3 bucket deletion ======
def block_bucket_deletion_policy(bucket_name):
    s3 = boto3.client("s3")
    policy = {
      "Version":"2012-10-17",
      "Statement":[{
        "Effect":"Deny",
        "Principal":"*",
        "Action":["s3:DeleteBucket","s3:DeleteObject","s3:DeleteObjectVersion"],
        "Resource":[f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"]
      }]
    }
    s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
    print("Deletion blocked.")


# ====== Launch SageMaker Notebook Instance ======
sagemaker_client = boto3.client("sagemaker")
def create_notebook_instance(name, instance_type, role_arn):
    try:
        sagemaker_client.create_notebook_instance(
            NotebookInstanceName=name,
            InstanceType=instance_type,
            RoleArn=role_arn,
        )
        print("Notebook launching...")
    except Exception as e:
        print("Exists / Error:", e)


# ====== ML Pipeline: Train Linear Regression ======
def train_linear_regression(df, target_col):
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X.fillna(0), y, test_size=0.2, random_state=42
    )

    model = LinearRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print("MSE:", mse)
    return model, mse


# ====== Add noise & compare ======
def add_noise_and_compare(df, target_col, noise_std=0.3):
    df2 = df.copy()
    numcols = df2.select_dtypes(include=[np.number]).columns

    df2[numcols] = df2[numcols] + np.random.normal(0, noise_std, size=df2[numcols].shape)

    _, mse1 = train_linear_regression(df, target_col)
    _, mse2 = train_linear_regression(df2, target_col)

    print(f"Original MSE = {mse1}, Noisy MSE = {mse2}")


# ====== Missing Value Handling ======
def impute_missing(df, strategy="median"):
    imputer = SimpleImputer(strategy=strategy)
    numcols = df.select_dtypes(include=[np.number]).columns
    df[numcols] = imputer.fit_transform(df[numcols])
    print("Missing values imputed with:", strategy)
    return df


# ====== Hyperparameter Tuning (Ridge) ======
def simple_hyperparam_tune(df, target_col):
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col]).fillna(0)
    y = df[target_col]

    gs = GridSearchCV(Ridge(), {"alpha": [0.01, 0.1, 1, 10]}, cv=3)
    gs.fit(X, y)
    print("Best params:", gs.best_params_)
    return gs.best_estimator_


# ====== Comprehend NLP ======
comprehend = boto3.client("comprehend")

def detect_language(text):
    return comprehend.detect_dominant_language(Text=text)

def detect_sentiment(text, lang='en'):
    return comprehend.detect_sentiment(Text=text, LanguageCode=lang)


# ====== Rekognition ======
rek = boto3.client("rekognition")

def detect_faces(bucket, key):
    return rek.detect_faces(Image={"S3Object": {"Bucket": bucket, "Name": key}}, Attributes=["ALL"])

def compare_faces(bucket, img1, img2):
    return rek.compare_faces(
        SourceImage={"S3Object": {"Bucket": bucket, "Name": img1}},
        TargetImage={"S3Object": {"Bucket": bucket, "Name": img2}},
        SimilarityThreshold=80
    )


# ====== SIMPLE CHATBOTS ======
def hotel_bot(msg): return "Provide dates for booking." if "book" in msg.lower() else "I handle hotel bookings."
def support_bot(msg): return "Provide order ID." if "refund" in msg.lower() else "Describe issue."
def product_bot(msg): return "Gaming or office laptop?" if "laptop" in msg.lower() else "Tell product category."
def health_bot(msg): return "Do you also have fever?" if "headache" in msg.lower() else "Mention symptoms."
def college_bot(msg): return "Which course?" if "admission" in msg.lower() else "Ask about fees/courses."
def restaurant_bot(msg): return "Date/time & people?" if "table" in msg.lower() else "I handle reservations."


# ====== MAIN EXECUTION EXAMPLE ======
if __name__ == "__main__":
    # Load your local house price dataset
    df = load_local_csv(YOUR_LOCAL_PATH)

    df = impute_missing(df)
    model, mse = train_linear_regression(df, target_col=df.columns[-1])
    add_noise_and_compare(df, df.columns[-1]()_
