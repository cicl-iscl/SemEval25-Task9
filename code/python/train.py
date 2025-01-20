import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from preprocessor import Datapreprocess  #


class LGBTrainer:
    def __init__(self, data_path, model_params=None):

        self.data_path = data_path
        self.data_processor = Datapreprocess()  # 初始化数据预处理类
        self.model_params = model_params if model_params else {}
        self.model = None

    def load_and_preprocess_data(self):


        self.data_processor.read_csv_file(self.data_path)

        self.data_processor.feature_extraction()
        X = self.data_processor.features

        y = self.data_processor.product_label

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):

        self.model = lgb.LGBMClassifier(**self.model_params)

        self.model.fit(X_train, y_train)
        print("Model training complete!")

    def evaluate_model(self, X_test, y_test):

        y_pred = self.model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy = accuracy_score(y_test, y_pred)

        print(f"F1 Score: {f1}")
        print(f"Accuracy: {accuracy}")
        return f1, accuracy

    def run(self):

        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()

        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)


if __name__ == "__main__":

    data_path = "Data/incidents_train.csv"

    model_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "boosting_type": "gbdt",
        "objective": "multiclass",
        "num_class": 5,
        "random_state": 42
    }

    trainer = LGBTrainer(data_path, model_params)

    trainer.run()
