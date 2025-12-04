# Single Jupyter cell: AWS + ML utility snippets (simple, one-line-per-step style)
# Install if needed: !pip install boto3 sagemaker scikit-learn pandas

import json, io, boto3, sagemaker, numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# ====== 1) Create IAM users with least-privilege (ML, Data, DevOps) ======
iam = boto3.client('iam')
def create_user_minimal(name):
    print("Create user (console may require MFA root for some ops).")
    try:
        iam.create_user(UserName=name)
    except Exception as e:
        print("Note:", e)
    # attach minimal managed policies or custom policies later
    return f"iam://{name}"

# ====== 2) IAM policy allowing only Amazon Comprehend ======
COMPREHEND_ONLY_POLICY = {
  "Version":"2012-10-17",
  "Statement":[
    {"Effect":"Allow","Action":["comprehend:*"],"Resource":"*"}
  ]
}
def create_comprehend_policy(name="ComprehendOnlyPolicy"):
    resp = iam.create_policy(PolicyName=name, PolicyDocument=json.dumps(COMPREHEND_ONLY_POLICY))
    return resp['Policy']['Arn']

# ====== 3) SageMaker role restricted to a single S3 bucket ======
def create_sagemaker_role_for_bucket(role_name, bucket_name):
    assume = {
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},
                   "Action":"sts:AssumeRole"}]
    }
    try:
        r = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume))
    except Exception as e:
        print("Role exists or error:", e)
        r = iam.get_role(RoleName=role_name)
    role_arn = r['Role']['Arn'] if isinstance(r, dict) else r['Role']['Arn']
    # Attach inline policy that allows S3 access only to the specified bucket
    s3_policy = {
      "Version":"2012-10-17",
      "Statement":[
        {"Effect":"Allow","Action":["s3:PutObject","s3:GetObject","s3:ListBucket"],
         "Resource":[f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"]}
      ]
    }
    iam.put_role_policy(RoleName=role_name, PolicyName=f"{role_name}-S3Access", PolicyDocument=json.dumps(s3_policy))
    return role_arn

# ====== 4) Policy that blocks S3 bucket deletion (bucket policy) ======
def block_bucket_deletion_policy(bucket_name):
    # Bucket policy denies s3:DeleteBucket and s3:DeleteObject for everyone except an explicit principal (optional)
    policy = {
      "Version":"2012-10-17",
      "Statement":[
        {"Sid":"DenyBucketAndObjectDeletion",
         "Effect":"Deny",
         "Principal":"*",
         "Action":["s3:DeleteBucket","s3:DeleteObject","s3:DeleteObjectVersion"],
         "Resource":[f"arn:aws:s3:::{bucket_name}", f"arn:aws:s3:::{bucket_name}/*"]}
      ]
    }
    s3 = boto3.client('s3')
    s3.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
    print("Bucket policy applied to block deletions (validate in console).")

# ====== 5) Launch SageMaker Notebook Instance (simple) ======
sagemaker_client = boto3.client('sagemaker')
def create_notebook_instance(name, instance_type, role_arn):
    try:
        sagemaker_client.create_notebook_instance(
            NotebookInstanceName=name,
            InstanceType=instance_type,
            RoleArn=role_arn,
            DirectInternetAccess='Enabled'
        )
        print("Notebook create request submitted.")
    except Exception as e:
        print("Error/exists:", e)

# ====== 6) Load dataset from mounted S3 into notebook (download to pandas) ======
s3 = boto3.client('s3')
def load_csv_from_s3(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print(f"Loaded {len(df)} rows from s3://{bucket}/{key}")
    return df

# ====== 7) Linear regression training on house-price dataset ======
def train_linear_regression(df, target_col, feature_cols=None):
    df = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c!= target_col]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    lm = LinearRegression().fit(X_train,y_train)
    preds = lm.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print("MSE:", mse)
    return lm, mse

# ====== 8) Add noise to dataset and compare accuracy (MSE) ======
def add_noise_and_compare(df, target_col, noise_std=0.1):
    dfn = df.copy()
    numcols = dfn.select_dtypes(include=[np.number]).columns.tolist()
    noise = np.random.normal(0, noise_std, size=dfn[numcols].shape)
    dfn[numcols] = dfn[numcols] + noise
    _, mse1 = train_linear_regression(df, target_col)
    _, mse2 = train_linear_regression(dfn, target_col)
    print(f"MSE original={mse1:.4f} noisy={mse2:.4f}")

# ====== 9) Handle missing values in dataset (SimpleImputer) ======
def impute_missing(df, strategy='median'):
    imp = SimpleImputer(strategy=strategy)
    numcols = df.select_dtypes(include=[np.number]).columns
    df[numcols] = imp.fit_transform(df[numcols])
    print("Imputed numeric columns with", strategy)
    return df

# ====== 10) Basic hyperparameter tuning (Ridge alpha via GridSearch) ======
def simple_hyperparam_tune(df, target_col):
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col])
    y = df[target_col]
    gs = GridSearchCV(Ridge(), {'alpha':[0.01,0.1,1,10]}, cv=3)
    gs.fit(X.fillna(0), y)
    print("Best params:", gs.best_params_, "best score:", gs.best_score_)
    return gs.best_estimator_

# ====== 11) Amazon Comprehend: detect language and sentiment analysis ======
comprehend = boto3.client('comprehend')
def detect_language(text):
    r = comprehend.detect_dominant_language(Text=text)
    return r.get('Languages',[])

def detect_sentiment(text, lang='en'):
    r = comprehend.detect_sentiment(Text=text, LanguageCode=lang)
    return r

def sentiment_trends(texts, lang='en'):
    # texts: list[str], returns simple counts/trend by sentiment
    counts = {'POSITIVE':0,'NEGATIVE':0,'NEUTRAL':0,'MIXED':0}
    for t in texts:
        r = comprehend.detect_sentiment(Text=t, LanguageCode=lang)
        counts[r['Sentiment']] += 1
    return counts

# ====== 12) Small sample rule-based chatbots (hotel booking, customer support, product rec, healthcare, college, restaurant) ======
def hotel_booking_bot(message):
    if "book" in message.lower() and "room" in message.lower():
        return "Sure — dates? Guests? I will check availability."
    return "I can help book rooms, check status or cancel. Tell me what you want."

def customer_support_bot(message):
    if "refund" in message.lower(): return "Please share order id to start refund."
    return "Describe your issue in one sentence."

def product_recommender_bot(message):
    if "laptop" in message.lower(): return "Prefer portability or power?"
    return "Tell me product category and budget."

def healthcare_symptom_bot(message):
    if "headache" in message.lower(): return "Have you taken any meds? Do you have fever?"
    return "I can suggest general guidance, not a diagnosis."

def college_enquiry_bot(message):
    if "admission" in message.lower(): return "Admissions open — which program?"
    return "Ask about courses, fees, or deadlines."

def restaurant_reservation_bot(message):
    if "reserve" in message.lower() or "table" in message.lower(): return "For which date/time and how many people?"
    return "I can reserve a table or show menus."

# ====== 13) Amazon Rekognition: detect faces + compare two images ======
rek = boto3.client('rekognition')
def detect_faces_attributes(bucket, key):
    resp = rek.detect_faces(Image={'S3Object':{'Bucket':bucket,'Name':key}}, Attributes=['ALL'])
    return resp['FaceDetails']  # list of faces with attributes

def compare_two_images(bucket, source_key, target_key, threshold=80):
    resp = rek.compare_faces(SourceImage={'S3Object':{'Bucket':bucket,'Name':source_key}},
                             TargetImage={'S3Object':{'Bucket':bucket,'Name':target_key}},
                             SimilarityThreshold=threshold)
    return resp['FaceMatches']

# ====== Quick example usage (change these placeholders) ======
if __name__ == "__main__":
    # 1) Create users (example names)
    for u in ['ml-user','data-user','devops-user']:
        create_user_minimal(u)
    # 2) Create Comprehend-only policy (returns ARN)
    try:
        arn = create_comprehend_policy()
        print("Comprehend policy ARN:", arn)
    except Exception as e:
        print("Policy create/exists:", e)
    # 3) Role for SageMaker restricted to bucket
    # role_arn = create_sagemaker_role_for_bucket("SageRoleForBucket", "YOUR_BUCKET")
    # print("SageMaker role:", role_arn)
    # 4) Block deletion (apply to bucket)
    # block_bucket_deletion_policy("YOUR_BUCKET")
    # 5) Create notebook instance (requires valid role ARN)
    # create_notebook_instance("my-notebook", "ml.t2.medium", role_arn)
    # 6) Loading dataset (example)
    # df = load_csv_from_s3("YOUR_BUCKET","datasets/house_prices.csv")
    # 7) Preprocess and train
    # df = impute_missing(df)
    # model, mse = train_linear_regression(df, target_col="price")
    # 8) Add noise and compare
    # add_noise_and_compare(df, "price", noise_std=0.5)
    # 9) Hyperparameter tuning
    # best_model = simple_hyperparam_tune(df, "price")
    # 10) Comprehend
    # print(detect_language("Bonjour tout le monde"))
    # print(detect_sentiment("I love this product!", lang='en'))
    # 11) Sentiment trends example
    # print(sentiment_trends(["I love it","It is bad","Neutral comment"], lang='en'))
    # 12) Chatbot example
    print(hotel_booking_bot("Please book a room for 2 adults"))
    # 13) Rekognition examples (must have images in S3)
    # print(detect_faces_attributes("YOUR_BUCKET","images/person.jpg"))
    # print(compare_two_images("YOUR_BUCKET","images/a.jpg","images/b.jpg"))
    print("Done. Replace placeholders and uncomment the lines to run AWS actions.")
