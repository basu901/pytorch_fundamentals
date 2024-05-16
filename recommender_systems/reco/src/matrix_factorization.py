import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import time
import matplotlib.pyplot as plt

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from pathlib import Path



class RatingDataSetLoader(Dataset):
    def __init__(self, data_path, num_factors):
        #Reading the whole csv file into memory
        self.data = pd.read_csv(data_path, header=None, sep=",",names = ['user_id', 'item_id', 'rating', 'timestamp'])
        self.num_factors = num_factors
        self._reindex()
        
        

    def __getitem__(self,idx):
        data = self.data.iloc[idx]
        movie = self.embedding_user(torch.tensor(self.movie_to_id[data["item_id"]]))
        user = self.embedding_user(torch.tensor(self.user_to_id[data["user_id"]]))

        # we are explicitly specifying the dtype for rating as the loss function expects a float and not Long.
        rating = torch.tensor(data["rating"],dtype=torch.float32) 
        return (user, movie, rating)

    def __len__(self):
        return len(self.data)

    def _reindex(self):
        users = (self.data["user_id"].drop_duplicates())
        self.user_to_id = {w: i for i, w in enumerate(users,0)}
        #self.data["user_id"] = self.data["user_id"].apply(lambda x: self.user_to_id[x])

        movies = (self.data["item_id"].drop_duplicates())
        self.movie_to_id = {w: i for i, w in enumerate(movies,0)}
        #self.data["item_id"] = self.data["item_id"].apply(lambda x: self.movie_to_id[x])

        self.embedding_user = nn.Embedding(len(self.user_to_id), embedding_dim = self.num_factors)
        self.embedding_user = nn.Embedding(len(self.movie_to_id), embedding_dim = self.num_factors)

        

class Model(nn.Module):
    def __init__(self, input_features, h1=10, h2=10):
        super().__init__()
        self.fc1 = nn.Linear(input_features,h1)
        self.fc2 = nn.Linear(h1, h2)
        self.output_layer = nn.Linear(h2, 1)

    def forward(self,input_features_users, input_features_movies):
        m_input = torch.cat((input_features_users, input_features_movies), 1)
        x = F.relu(self.fc1(m_input))
        x = F.relu(self.fc2(x))
        pred_rating = F.relu(self.output_layer(x))
        return pred_rating


def create_csv(data_source, csv_source):
    data_rows = list()
    with open(str(data_source),"r") as f:
        data = f.readlines()
        for row in data:
            data_rows.append(row.strip().split())

    with open(str(csv_source),"w") as csv_path:
        write_files = csv.writer(csv_path)
        write_files.writerows(data_rows)


if __name__=="__main__":
    data_source = Path("/Users/shaunakbasu/Documents/Datasets/ml-100k/ml-100k/u.data")
    csv_source = Path("/Users/shaunakbasu/Documents/Datasets/ml-100k/ml-100k/u_data.csv")

    if not csv_source.exists():
        create_csv(data_source, csv_source)
    
    num_factors = 4
    dataset = RatingDataSetLoader(csv_source, num_factors)
    data_loader = DataLoader(dataset, shuffle=True, batch_size=5)

    #Since we are stacking both the user embedding and the movie embedding, we multiply the num_factors by 2
    m = Model(num_factors*2)

    epochs = 25

    #Set the criterion for measuring the loss function
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    #Choose Adam Optimizer, learning rate=lr; if the error doesn't decrease a number of epochs, lower the learning rate(Adam property?)
    optimizer=torch.optim.Adam(m.parameters(),lr=0.01)
    loss_list = list()

    start = time.perf_counter()

    for i in range(0, epochs):
        epoch_loss = 0
        counter = 0
        for idx, data in enumerate(data_loader,1):
            user_embed = data[0]
            movie_embed = data[1]
            expected_ratings = data[2]
            predicted_rating = m.forward(user_embed,movie_embed).squeeze(-1)
            loss = criterion(expected_ratings,predicted_rating)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()
            counter += 1
        loss_list.append(epoch_loss/counter)
        print(f"Epoch: {i}, loss: {epoch_loss/counter}")

    stop = time.perf_counter()

    print(f"Total time taken: {stop-start}s")
    plt.plot(range(epochs),loss_list)
    plt.ylabel("loss/error")
    plt.xlabel("Epoch")
    plt.show()
    
    #Total time taken: 579.0461256660055s
    #Thi above result was for  25 epochs

    


