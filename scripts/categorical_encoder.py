import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def process_categorical_features(df, columns, encoding_type='auto'):
    """
    Обработка категориальных фичей
    
    Параметры:
    -----------
    df : pd.Dataframe
      Датафрейм
    columns : str or list
        Название столбца или список столбцов для обработки
    encoding_type : str or dict, optional ('auto', 'onehot', 'label')
        Тип кодирования:
        - 'auto' - автоматический выбор (onehot для <=15 уникальных значений, иначе label)
        - 'onehot' - one-hot encoding для всех столбцов
        - 'label' - label encoding для всех столбцов         
    """
    df_processed = df.copy()
    encoding_info = {}
    
    if isinstance(columns, str):
        columns = [columns]
    if not isinstance(encoding_type, dict):
        encoding_type = {col: encoding_type for col in columns}
    
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Столбцы не найдены в DataFrame: {missing_columns}")
    
    for column in columns:
        col_info = {
            'column': column,
            'original_type': str(df[column].dtype),
            'encoding_type': None,
            'categories': None
        }
        
        unique_values = df[column].nunique()
        is_categorical = (df[column].dtype == 'object') or (unique_values < 20)
        
        if not is_categorical:
            col_info['encoding_type'] = 'none'
            encoding_info[column] = col_info
            continue
        
        current_encoding = encoding_type.get(column, 'auto')
        
        if current_encoding == 'auto':
            if unique_values <= 15:
                current_encoding = 'onehot'
            else:
                current_encoding = 'label'
        
        if current_encoding == 'onehot':
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = ohe.fit_transform(df_processed[[column]])
            
            categories = ohe.categories_[0][1:]
            new_columns = [f"{column}_{cat}" for cat in categories]
            
            df_processed.drop(columns=[column], inplace=True)
            df_processed[new_columns] = encoded_data
            
            col_info['encoding_type'] = 'onehot'
            col_info['categories'] = categories.tolist()
            
        elif current_encoding == 'label':
            le = LabelEncoder()
            df_processed[column] = le.fit_transform(df_processed[column])
            
            col_info['encoding_type'] = 'label'
            col_info['categories'] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        else:
            raise ValueError(f"Неподдерживаемый тип кодирования для столбца {column}. ")
        
        encoding_info[column] = col_info
    
    return df_processed, encoding_info


"""
Examples
df_processed, info = process_categorical_features(df, ['gender', 'city'], 'onehot')

df_processed, info = process_categorical_features(
    df, 
    ['gender', 'city'], 
    {'gender': 'onehot', 'city': 'label'}
)

df_processed, info = process_categorical_features(df, 'gender', 'onehot')
"""