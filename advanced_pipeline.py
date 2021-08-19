import seaborn as sns
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV

titanic = sns.load_dataset('titanic')
print(titanic.dtypes)
print(titanic.iloc[:5].to_markdown())

X = titanic.drop("survived", axis=1)
y = titanic["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

pca = PCA(n_components = 1)

numeric_features = ["pclass", "age", "fare"]
categorical_features = ["sex", "deck", "alone"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

rf = Pipeline(steps=[('preprocessor', preprocessor),
                     ("pca", pca),
                     ('classifier', RandomForestClassifier())])

rf.fit(X_train, y_train)
res = rf.predict(X_test)
score = rf.score(X_test, y_test)


transformer = make_column_transformer((StandardScaler(), ["age", "fare"]))

model = make_pipeline(transformer, SGDClassifier())
model.fit(X, y)
print("oik")
