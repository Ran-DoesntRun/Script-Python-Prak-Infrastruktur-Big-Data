#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install Kagglehub
pip install kagglehub

import kagglehub

# Mendownload dataset dari kagglehub
path = kagglehub.dataset_download("chaitanyahivlekar/large-movie-dataset")

# Melihat path tempat dataset
print(path)


# In[22]:


Ukuran File DataSet
print(os.stat('movies_dataset.csv').st_size/(1024 * 1024))


# In[1]:


# Import Libraries
from pyspark.sql.types import StructType, StructField, FloatType, BooleanType
from pyspark.sql.functions import col, sqrt, lit
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark import SQLContext

# Setup the Configuration
conf = pyspark.SparkConf()

spark = SparkSession.builder.appName("Pengujian Spark").master("local").getOrCreate()
sqlcontext = SQLContext(sc)


# In[2]:


#MEMBACA DATA
data = spark.read.csv('movies_dataset.csv', inferSchema=True, header=True)

#MEMBUAT ID UNTUK SETIAP MOVIE
id_movie = data.select("Movie_Name").distinct().withColumn("Movie_Id", monotonically_increasing_id())
id_movie = id_movie.filter(id_movie.Movie_Id <= 2147483647)

#MENGGABUNGKAN KOLOM MOVIE ID DENGAN DATASET ORI
gabung = data.join(id_movie, on="Movie_Name", how="inner")

#DATASET 3 KOLOM UTAMA UNTUK DIPROSES
final_data = gabung.select('User_Id','Movie_Id','Rating')


# In[3]:


final_data.count()


# In[3]:


#MENGECEK SKEMA KOLOM DATA
final_data = final_data.sort('User_Id')
final_data.printSchema()


# In[3]:


#MELAKUKAN PERUBAHAN SKEMA DATA
final_data = final_data.withColumn('User_Id', final_data['User_Id'].cast('int')).withColumn('Movie_Id', final_data['Movie_Id'].cast('int')).withColumn('Rating', final_data['Rating'].cast('float'))
final_data.printSchema()


# In[4]:


#MEMBAGI RASIO DATA UNTUK TRAINING, VALIDASI DAN TEST
train, validasi, test = final_data.randomSplit([0.7,0.20,0.10], seed = 0)
print("The number of ratings in each set: {}, {}, {}".format(train.count(), validasi.count(), test.count()))


# In[5]:


# FUNGSI UNTUK MENGHITUNG RMSE 
def RMSE(predictions):
    squared_diff = predictions.withColumn("squared_diff", pow(col("rating") - col("prediction"), 2))
    mse = squared_diff.selectExpr("mean(squared_diff) as mse").first().mse
    return mse ** 0.5


# In[6]:


#FUNGSI UNTUK MENENTUKAN MODEL TERBAIK

def GridSearch(train, valid, num_iterations, reg_param, n_factors):
    min_rmse = float('inf')
    best_n = -1
    best_reg = 0
    best_model = None
    
    for n in n_factors:
        for reg in reg_param:
            als = ALS(rank = n, 
                      maxIter = num_iterations, 
                      seed = 0, 
                      regParam = reg,
                      userCol="User_Id", 
                      itemCol="Movie_Id", 
                      ratingCol="Rating", 
                      coldStartStrategy="drop")            
            model = als.fit(train)
            predictions = model.transform(valid)
            rmse = RMSE(predictions)     
            print('{} latent factors and regularization = {}: validation RMSE is {}'.format(n, reg, rmse))
            # track the best model using RMSE
            if rmse < min_rmse:
                min_rmse = rmse
                best_n = n
                best_reg = reg
                best_model = model
                
    pred = best_model.transform(train)
    train_rmse = RMSE(pred)
    # best model and its metrics
    print('\nThe best model has {} latent factors and regularization = {}:'.format(best_n, best_reg))
    print('traning RMSE is {}; validation RMSE is {}'.format(train_rmse, min_rmse))
    return best_model


# In[7]:


# MEMBUAT MODEL DENGAN MENGGUNAKAN FUNGSI GRID SEARCH
import time
num_iterations = 10
ranks = [10, 20]
reg_params = [0.01, 0.05, 0.1]


start_time = time.time()
final_model = GridSearch(train, validasi, num_iterations, reg_params, ranks)
print('Total Runtime: {:.2f} seconds'.format(time.time() - start_time))


# In[8]:


# MELAKUKAN TES AKURASI MODEL TERHADAP DATA TEST MENGGUNAKAN PERHITUNGAN RMSE
pred_test = final_model.transform(test)
print('The testing RMSE is ' + str(RMSE(pred_test)))


# In[9]:


# MELAKUKAN TEST UNTUK SATU USER
single_user = test.filter(test['User_Id']==2).select(['Movie_Id','User_Id'])
single_user.show()


# In[10]:


# MELAKUKAN VERIFIKASI PREDIKSI RATING
reccomendations = final_model.transform(single_user)
reccomendations.orderBy('prediction',ascending=False).show()


# In[11]:


# MEMUNCULKAN NAMA FILM BERDASARKAN ID
reccomendations.join(id_movie, reccomendations.Movie_Id == id_movie.Movie_Id, 'inner').show()


# In[12]:


# MEMILIH SATU USER DARI DATA YANG DISIMPAN UNTUK DITES
user_id = 12
single_user_ratings = test.filter(test['User_Id'] == user_id).select(['Movie_Id','User_Id', 'Rating'])

# MENAMPILKAN RATING FILM YANG TELAH DIBERIKAN OLEH USER
print("Movies liked by user with ID", user_id)
single_user_ratings.join(id_movie, 'Movie_Id').select('Movie_Id', 'Movie_Name', 'Rating').show()

# MELAKUKAN GENERATE REKOMENDASI
all_movies = id_movie.select('Movie_Id')
user_movies = single_user_ratings.select('Movie_Id').distinct()
movies_to_recommend = all_movies.subtract(user_movies)

# MEMPREDIKSI RATING FILM YANG AKAN DIREKOMENDASIKAN KEPADA USER
recommendations = final_model.transform(movies_to_recommend.withColumn('User_Id', lit(user_id)))

# MENGELEMINASI FILM YANG DIPREDIKSI TIDAK DISUKAI USER
recommendations = recommendations.filter(col('prediction') > 0)

# MENAMPILKAN FILM YANG DIREKOMENDASI
print("Recommended movies for user with ID", user_id)
recommended_movies = recommendations.join(id_movie, 'Movie_Id').select('Movie_Id', 'Movie_Name', 'prediction')

# MENGURUTAKN REKOMENDASI FILM BERDASARKAN PREDIKSI RATING
ordered_recommendations = recommended_movies.orderBy(col('prediction').desc())

# MENAMPILKAN REKOMENDASI FILM YANG TELAH DIURUTKAN
ordered_recommendations.show()


# In[ ]:




