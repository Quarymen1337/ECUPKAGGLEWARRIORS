import pandas as pd
def generate_history_matrix(dataframe, col_item_name, col_user_name):
    """
    Преобразование сырых данных
    Создание матрицы истории каждого пользователя
    Каждая строка это история покупок пользователя
    """
    user_item_matrix = {}
    df = dataframe.reset_index() 
    
    for idx, row in df.iterrows():
        user_id = row[col_user_name]
        history = row[col_item_name]
        
        aid_list = list({item["aid"] for item in history})
        
        user_item_matrix[user_id] = aid_list
    
    result_df = pd.DataFrame.from_dict(user_item_matrix, orient="index")
    
    result_df.index.name = "user_id"         
    result_df.columns.name = "item_id"       
    result_df = result_df.fillna(-1)  
    
    result_df.columns = [f"item_id_{i}" for i in range(len(result_df.columns))]
    return result_df


def prepare_data(df):
    """Преобразуем сырые данные в формат (user_id, item_id, timestamp)"""
    records = []
    for _, row in df.iterrows():
        user_id = row["session"]
        for event in row["events"]:
            records.append({
                "user_id": user_id,
                "item_id": event["aid"],
                "timestamp": event["ts"],
                "relevance": 1 
            })
    return pd.DataFrame(records)
    