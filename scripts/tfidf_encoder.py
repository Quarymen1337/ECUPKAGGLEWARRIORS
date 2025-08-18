import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'&#?\w+;', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = text.split() 
    tokens = [word for word in tokens if word not in stop_words] 
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def encode_text_with_tfidf(
    dataframe, 
    text_column, 
    max_features=60, 
    min_df=5, 
    max_df=0.85, 
    ngram_range=(5, 12),
    drop_original=True,
    drop_cleaned=True
):
    """    
    Параметры:
    ----------
    dataframe : pd.DataFrame
        Исходный датафрейм
    text_column : str
        Название столбца с текстом для обработки
    max_features : int, optional (default=60)
        Максимальное количество признаков TF-IDF
    min_df : int or float, optional (default=5)
        Минимальная частота слова для учета
    max_df : float, optional (default=0.85)
        Максимальная частота слова для учета (доля документов)
    ngram_range : tuple, optional (default=(5, 12))
        Диапазон n-грамм для извлечения
    drop_original : bool, optional (default=True)
        Удалять ли исходный текстовый столбец
    drop_cleaned : bool, optional (default=True)
        Удалять ли промежуточный столбец с очищенным текстом
    """
    df = dataframe.copy()
    cleaned_column = f'cleaned_{text_column}'
    df[cleaned_column] = df[text_column].apply(preprocess_text)
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )
    tfidf_matrix = tfidf.fit_transform(df[cleaned_column])
    tfidf_features = tfidf.get_feature_names_out()
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{text_column}_{feat}' for feat in tfidf_features]
    )
    
    result_df = pd.concat([df, tfidf_df], axis=1)
    
    if drop_original:
        result_df.drop(columns=[text_column], inplace=True)
    if drop_cleaned:
        result_df.drop(columns=[cleaned_column], inplace=True)
    
    return result_df