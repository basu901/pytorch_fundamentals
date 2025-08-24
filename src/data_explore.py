import os
import argparse
from typing import Dict
import pandas as pd
from pathlib import Path
from psycopg import connect
from dotenv import load_dotenv

load_dotenv()

db = os.environ['POSTGRES_DB_NAME']
pg_user = os.environ['POSTGRES_USERNAME']
pg_passwd = os.environ["POSTGRES_PASSWORD"]
pg_host = os.environ["POSTGRES_HOST"]
pg_port = os.environ["POSTGRES_PORT"]


def create_tables():
    """
    Creates the tables required:
    avg_rating: Stores the bias of each user, by calculating the avg rating for all movies viewed by the user
    movie_details: The master data which contains the rating given for each movie  by a particular user. 
    user_weights: Stores the weights between two users
    
    """
    with connect(f"dbname={db} user={pg_user} password={pg_passwd} host={pg_host} port={pg_port}") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS avg_rating(
                        id serial PRIMARY KEY,
                        user_id int,
                        rating_avg real)
            """)

            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS movie_details(
                            id serial PRIMARY KEY,
                            movie_id int,
                            user_id int,
                            rating real)
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_weights(
                            id serial PRIMARY KEY,
                            user_id_1 int,
                            user_id_2 int,
                            weight real)
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_movie_predicted(
                            id serial PRIMARY KEY,
                            user_id int,
                            movie_id int,
                            predicted_rating real)
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS movie_info(
                            id serial PRIMARY KEY,
                            user_id int,
                            movie_id int,
                            predicted_rating real)
                """)



def populate_movie_table(connection_params:Dict):
    """
    Populates the master table movie_data from the csv data.
    
    """
    from io import StringIO
    cur_dir = Path(__file__).cwd()
    dataset = cur_dir.joinpath("data","ml-100k","u_data.csv")
    df = pd.read_csv(str(dataset), header=None, names=['user_id','item_id','rating','timestamp'])

    #Re-ordering to match the movie_data schema
    df_movie_details = df.iloc[:,[1,0,2]]
    
    buffer = StringIO()
    df_movie_details.to_csv(buffer,header=False,index=True)
    buffer.seek(0)

    with connect(**connection_params) as conn:
    #with connect(f"dbname={db} user={pg_user} password={pg_passwd} host={pg_host} port={pg_port}") as conn:
        with conn.cursor() as cur:
            #The COPY command does not generate PRIMARY KEY automatically, hence "index=True" has been set when pushing to buffer
            with cur.copy("COPY movie_details FROM STDIN WITH (FORMAT CSV)" ) as copy:
                while data := buffer.read():
                    copy.write(data)



def calculate_avg_rating(user_id:tuple, connection_params:Dict):
    """
    Calculates the average rating/bias for each user present in user_id
    """
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating FROM movie_details WHERE user_id=%s", user_id)
            ratings = cur.fetchall()
            
    num_of_ratings = len(ratings)
    total_rating = 0
    
    for rating in ratings:
        total_rating += rating[0]

    return round(total_rating/num_of_ratings,2)


    



def populate_avg_rating_table(connection_params:Dict):
    """
    Find average rating for each user
    """

    avg_rating = {}

    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT(user_id) FROM movie_details")
            user_ids = cur.fetchall()

    for user in user_ids:
        avg_rating[user[0]] = calculate_avg_rating(user, connection_params)

    def push_average_ratings(avg_rating:tuple):
        with connect(**connection_params) as conn:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO avg_rating (user_id, rating_avg) VALUES (%s, %s)", avg_rating) 

    push_average_ratings(tuple(avg_rating.items()))




def find_users_and_rating_for_movie(movie_id:int, connection_params:Dict):
    """
    Find the list of users alongwith their ratings for a particular movie
    """
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, rating  FROM movie_details WHERE movie_id=%s", (movie_id,))
            rows = cur.fetchall()
    
    return rows




def ratings_from_common_movies(user_j:int, user_i:int, connection_params:Dict):
    """
    Find the set of movies between two users and return the ratings given by both users, for each of the movies.
    """
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT a.rating, b.rating FROM movie_details AS a JOIN movie_details AS b
                        ON a.movie_id = b.movie_id

                        WHERE a.user_id = %s AND b.user_id = %s 
                        """, (user_j, user_i))
            rows = cur.fetchall()
    return rows





def fetch_average(user_id:int, connection_params:Dict):
    """
    Fetch avg rating for a particular user
    """
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating_avg FROM avg_rating WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
    return row[0]




def find_weights(user_i:int, user_j:int, connection_params:Dict):
    """
    Fetch the weight between two users
    """
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT weight FROM user_weights WHERE user_id_1=%s AND user_id_2=%s", (min(user_i, user_j), max(user_i, user_j)))
            row = cur.fetchone()
    if row:
        return row[0]
    else:
        return None




def insert_weights(user_i:int, user_j:int, weight:float, connection_params:Dict):
    """
    Insert weight between two users into user_weights
    """
    with connect(**connection_params) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO user_weights (user_id_1, user_id_2, weight) VALUES (%s, %s, %s)", (min(user_i, user_j), max(user_i, user_j), weight)) 





def predict_rating(user_id:int, movie_id:int, connection_params:Dict):
    """
    Predict the rating for a particular pair of (user, movie)
    """
    import math

    #Check if the user already has a rating for the movie
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating FROM movie_details WHERE movie_id=%s AND user_id=%s", (movie_id, user_id))
            row = cur.fetchone()
            if row:
                return f"Prediction Not Required. Rating is {row[0]}"
            
            cur.execute("SELECT predicted_rating FROM user_movie_predicted WHERE user_id=%s AND movie_id=%s",(user_id,movie_id))
            row = cur.fetchone()
            if row:
                return row[0]

    #user_j = has seen movie_id and has movies in common with user_id
    #Stores info for user_j.Stores the weight, rating given for movie_id, avg_rating
    user_j_info = list()

    #The bias of user_id
    avg_user_id = fetch_average(user_id, connection_params)

    #Find the list of users who have seen this movie       
    user_rating_list = find_users_and_rating_for_movie(movie_id, connection_params)

    for item in user_rating_list:
        user = item[0]
        rating_j = item[1]
        avg_j = fetch_average(user, connection_params)

        #Check if weights already exist
        weight = find_weights(user_id, user, connection_params)
        
        if weight:
            user_j_info.append((user, rating_j, avg_j, weight))
        else:
            #Calculate and Store weights
            #common_movies[i][0] refers to ratings for user_id
            #common_movies[i][1] refers to ratings for user
            
            common_movies = ratings_from_common_movies(user_id, user, connection_params)
            ratings_df = pd.DataFrame(common_movies, columns=["user_id_rating", "user_rating"])
            ratings_df['user_id_rating'] = ratings_df["user_id_rating"] - avg_user_id
            ratings_df['user_rating'] = ratings_df["user_rating"] - avg_j
            ratings_df['user_user_rating'] = ratings_df['user_id_rating'] * ratings_df['user_rating']
            ratings_df['user_id_rating_squared'] = ratings_df["user_id_rating"]**2
            ratings_df['user_rating_squared'] = ratings_df["user_rating"]**2
            # print(ratings_df)
            # input()
            weight_params = ratings_df.sum(axis=0)
            #print(weight_params)
            weight = (weight_params['user_user_rating']/(math.sqrt(weight_params['user_id_rating_squared'])*math.sqrt(weight_params['user_rating_squared'])))
            # print(weight)
            # text = input()
            # if text=="exit":
            #     return

            insert_weights(user_id, user, weight, connection_params)
            user_j_info.append((user, rating_j, avg_j, weight))

    #Iterate through user_j_info to predict the final rating
    weighted_rating = 0
    total_weights = 0
    for item in user_j_info:
        rating_given = item[1]
        bias = item[2]
        user_weight = item[3]
        weighted_rating += user_weight*(rating_given-bias)
        total_weights += abs(user_weight)

    predicted_rating = round(avg_user_id + (weighted_rating/total_weights),2)

    #Store predicted rating for subsequent calls
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_movie_predicted (user_id, movie_id, predicted_rating) VALUES(%s,%s,%s)", (user_id, movie_id, predicted_rating))
    

    return predicted_rating
    

if __name__=="__main__":
    connection_params ={
        "dbname":f"{db}",
        "user":f"{pg_user}",
        "password":f"{pg_passwd}", 
        "host":f"{pg_host}",
        "port":f"{pg_port}"
    }

    #Uncomment the following three lines, on the first pass of the code
    #create_tables()
    #populate_tables()
    #populate_avg_rating_table(connection_params)

    parser = argparse.ArgumentParser()
    parser.add_argument("-u",'--user', type=int,required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--movie_id', type=int, help='Movie ID, should be int')
    group.add_argument('-mn', '--movie_name',type=str, help='Movie Name')

    args = parser.parse_args()

    print(predict_rating(args.user, args.movie_id, connection_params))