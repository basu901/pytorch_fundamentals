import os
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
    with connect(f"dbname={db} user={pg_user} password={pg_passwd} host={pg_host} port={pg_port}") as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE avg_rating(
                        id serial PRIMARY KEY,
                        user_id int,
                        rating_avg real)
            """)

            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE movie_details(
                            id serial PRIMARY KEY,
                            movie_id int,
                            user_id int,
                            rating real)
                """)

                cur.execute("""
                    CREATE TABLE user_weights(
                            id serial PRIMARY KEY,
                            user_id_1 int,
                            user_id_2 int,
                            weight real)
                """)



def populate_movie_table(connection_params:Dict):
    from io import StringIO
    cur_dir = Path(__file__).cwd()
    dataset = cur_dir.joinpath("data","ml-100k","u_data.csv")
    df = pd.read_csv(str(dataset), header=None, names=['user_id','item_id','rating','timestamp'])
    df_movie_details = df.iloc[:,[1,0,2]]
    print(df_movie_details.head(3))
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
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating FROM movie_details WHERE user_id=%s", user_id)
            ratings = cur.fetchall()
            
    num_of_ratings = len(ratings)
    total_rating = 0
    
    for rating in ratings:
        total_rating += rating[0]

    return round(total_rating/num_of_ratings,2)




def push_average_ratings(avg_rating:tuple, connection_params:Dict):
    
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO avg_rating (user_id, rating_avg) VALUES (%s, %s)", avg_rating) 
    



def populate_avg_rating_table(connection_params:Dict):

    avg_rating = {}

    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT(user_id) FROM movie_details")
            user_ids = cur.fetchall()

    for user in user_ids:
        avg_rating[user[0]] = calculate_avg_rating(user, connection_params)

    push_average_ratings(tuple(avg_rating.items()), connection_params)


def find_users_for_movie(movie_id, connection_params):

    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, rating  FROM movie_details WHERE movie_id=%s", (movie_id))
            rows = cur.fetchall()
    
    return rows


def list_of_common_movies(user_j, user_i, connection_params):
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                        SELECT a.rating, b.rating FROM movie_details AS a JOIN movie_details AS b
                        ON a.movie_id = b.movie_id

                        WHERE a.user_id = %s AND b.user_id = %s 
                        """, (user_j, user_i))
            row = cur.fetchone()
            if row:
                return f"Prediction Not Required. Rating is {row[0]}"



def fetch_average(user_id, connection_params):
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating_avg FROM avg_rating WHERE user_id=%s", (user_id))
            row = cur.fetchone()
    return row[0]



def find_weights(user_i, user_j, connection_params):
    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT weight FROM user_weights WHERE user_id_1=%s AND user_id_2=%s", (min(user_i, user_j), max(user_i, user_j)))
            row = cur.fetchone()
    if row:
        return row[0]
    else:
        return None


def predict_rating(user_id, movie_id, connection_params):

    with connect(**connection_params) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT rating FROM movie_details WHERE movie_id=%s AND user_id=%s", (movie_id, user_id))
            row = cur.fetchone()
            if row:
                return f"Prediction Not Required. Rating is {row[0]}"

    #This list will contain what the weight is for the user, rating given by the user, avg_rating given by the user
    user_j_info = list()

    #Find the list of users who have seen this movie       
    user_rating_list = find_users_and_rating_for_movie(movie_id, connection_params)

    for item in user_rating_list:
        user = item[0]
        rating_j = item[1]
        avg_j = fetch_average(user, connection_params)

        #Check if weights already exist
        weights = find_weights(user_id, user, connection_params)
        
        if weights:
            user_j_info.append((user, rating_j, avg_j, weights))
        else:
            #Calculate and Store weights
            common_movies = list_of_common_movies(user_id, user, connection_params)



if __name__=="__main__":
    connection_params ={
        "dbname":f"{db}",
        "user":f"{pg_user}",
        "password":f"{pg_passwd}", 
        "host":f"{pg_host}",
        "port":f"{pg_port}"
    }

    #populate_tables()
    populate_avg_rating_table(connection_params)