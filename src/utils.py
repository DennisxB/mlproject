#model training
from sklean.linear_models import LinearRegression
from sklean.linear_models import Ridge, Lasso

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    #"XGBRegressor": XGBRegressor(), 
    #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
    }

# define X_data, y_data

def train_step(models, X_train, X_test, y_train, y_test):
    
    model_list = []
    r2_list =[]

    for i in tqdm(range(len(list(models)))):
        model = list(models.values())[i]
        model.fit(X_train, y_train) # Train model

        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate Train and Test dataset
        model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

        model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

        
        print(list(models.keys())[i])
        model_list.append(list(models.keys())[i])
        
        print('Model performance for Training set')
        print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
        print("- R2 Score: {:.4f}".format(model_train_r2))

        print('----------------------------------')
        
        print('Model performance for Test set')
        print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
        print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
        print("- R2 Score: {:.4f}".format(model_test_r2))
        r2_list.append(model_test_r2)
        
        print('='*35)
        print('\n')
        model_list = []


### data ingestion
import pandas as pd
def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv('data/stud.csv')

    return df

########### transformation ###########
# check and remove missing values

def remove_outliers(feature: pd.Series, lower_percentile: float, upper_percentile: float) -> pd.Series:
    lower_thresh = feature.quantile(lower_percentile)
    upper_thresh = feature.quantile(upper_percentile)

    feature = feature.map(lambda val: \
                          lower_thresh if val < lower_thresh  
                          else upper_thresh if val > upper_thresh \
                          else val)
    
    return feature 

def clean_data(df: pd.DataFrame, lower_percentile: float, upper_percentile: float) -> pd.DataFrame:
    df = df.fillna(df.mean(numeric_only=True))

    for col in df.columns:
        df[col] = remove_outliers(df[col], lower_percentile, upper_percentile)
    
    return df

    cap_outliers
    # remove outliers 

