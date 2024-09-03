import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

# Function to fetch data from CSV files
def fetch_data():
    try:
        # Define the file paths
        path = "C:/Users/Admin/Desktop/Data Sets/"
        file_projects = path + "projects.csv"
        file_milestones = path + "milestones.csv"
        file_tasks = path + "tasks.csv"
        
        # Read CSV files into pandas DataFrames
        df_projects = pd.read_csv(file_projects)
        df_milestones = pd.read_csv(file_milestones)
        df_tasks = pd.read_csv(file_tasks)
        
        return df_projects, df_milestones, df_tasks
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None

# Function to join the tables properly to avoid information loss
def join_tables(df_projects, df_milestones, df_tasks):
    # Join projects with milestones using project_id
    df = pd.merge(df_milestones, df_projects, on='project_id', how='left')
    
    # Join the result with tasks using project_id and milestone_id
    df = pd.merge(df_tasks, df, on=['project_id', 'milestone_id'], how='left')
    
    return df

# Function to handle date columns by extracting year, month, day, and duration
def process_dates(df):
    # List of date columns
    date_columns = ['start_date', 'end_date', 'milestone_start_date', 'milestone_end_date', 'task_start_date', 'task_due_date']

    # Loop through each date column and extract year, month, and day
    for col in date_columns:
        if col in df.columns:
            df[f'{col}_year'] = pd.to_datetime(df[col], errors='coerce').dt.year
            df[f'{col}_month'] = pd.to_datetime(df[col], errors='coerce').dt.month
            df[f'{col}_day'] = pd.to_datetime(df[col], errors='coerce').dt.day

    # Calculate durations
    if 'start_date' in df.columns and 'end_date' in df.columns:
        df['project_duration'] = (pd.to_datetime(df['end_date'], errors='coerce') - pd.to_datetime(df['start_date'], errors='coerce')).dt.days
    if 'milestone_start_date' in df.columns and 'milestone_end_date' in df.columns:
        df['milestone_duration'] = (pd.to_datetime(df['milestone_end_date'], errors='coerce') - pd.to_datetime(df['milestone_start_date'], errors='coerce')).dt.days
    if 'task_start_date' in df.columns and 'task_due_date' in df.columns:
        df['task_duration'] = (pd.to_datetime(df['task_due_date'], errors='coerce') - pd.to_datetime(df['task_start_date'], errors='coerce')).dt.days

    # Drop original date columns
    df = df.drop(columns=date_columns, errors='ignore')

    return df

# Function to encode categorical columns
def encode_and_drop_unwanted_columns(df):
    # Exclude columns that are not needed for correlation
    exclude_columns = ['project_id', 'project_name', 'milestone_name', 'milestone_id', 'task_id', 'task_name', 'owner']
    
    # Drop unnecessary columns
    df = df.drop(columns=exclude_columns, errors='ignore')

    # Process date columns
    df = process_dates(df)

    # Separate numeric and categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Apply label encoding to categorical columns with low cardinality (e.g., status)
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if df[col].nunique() <= 10:  # Assuming low unique values means ordinal columns
            df[col] = label_encoder.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df

# Function to drop highly correlated columns
def drop_highly_correlated(df, threshold=0.7):
    # Compute the correlation matrix (using Spearman method)
    corr_matrix = df.corr(method='spearman').abs()

    # Print the correlation matrix
    print("Correlation Matrix:")
    print(corr_matrix)

    # Select the upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find columns with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop these columns from the DataFrame
    df_dropped = df.drop(columns=to_drop)

    return df_dropped, to_drop

# Function to create and evaluate a RandomForestClassifier pipeline
def create_random_forest_pipeline(X, y):
    # Create a pipeline with scaling, PCA, and classifier
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', MinMaxScaler())
                ]), X.select_dtypes(include=['int64', 'float64']).columns),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), X.select_dtypes(include=['object']).columns)
            ])),
        ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
        ('classifier', RandomForestClassifier())
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search

# Function to print milestone_completion_mode for a given milestone
def print_milestone_completion_mode(milestone_id):
    try:
        # Define the file path
        path = "C:/Users/Admin/Desktop/Data Sets/milestones.csv"
        
        # Read the CSV file
        df_milestone = pd.read_csv(path)
        
        # Ensure milestone_id is quoted correctly in SQL query
        df_milestone_filtered = df_milestone[df_milestone['milestone_id'] == milestone_id]
        
        if not df_milestone_filtered.empty:
            print(f"Milestone Completion Mode for ID {milestone_id}: {df_milestone_filtered['milestone_completion_mode'].values[0]}")
        else:
            print(f"No milestone found with ID {milestone_id}")
    except Exception as e:
        print(f"Error fetching milestone completion mode: {e}")

# Main execution
if __name__ == "__main__":
    # Fetch and join data
    df_projects, df_milestones, df_tasks = fetch_data()
    if df_projects is not None and df_milestones is not None and df_tasks is not None:
        df = join_tables(df_projects, df_milestones, df_tasks)

        # Encode and drop unwanted columns
        df = encode_and_drop_unwanted_columns(df)
        
        # Drop highly correlated columns
        df, dropped_columns = drop_highly_correlated(df)
        print(f"Dropped columns due to high correlation: {dropped_columns}")

        # Split data into features and target variable
        X = df.drop('milestone_completion_mode', axis=1)
        y = df['milestone_completion_mode']

        # Create and evaluate RandomForestClassifier pipeline
        grid_search = create_random_forest_pipeline(X, y)

        # Print the best parameters and score
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_}")

        # Predict milestone_completion_mode for updated data
        milestone_id = 'zcr_749813000007488421'  # Replace with the milestone ID you want to predict (as a string)
        print_milestone_completion_mode(milestone_id)
