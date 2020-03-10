from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0)
        )
# load the iris dataset and split it into train and test sets
x, y = load_iris(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# fit the whole pipeline
pipe.fit(x_train, y_train)

print(accuracy_score(pipe.predict(x_test), y_test))
