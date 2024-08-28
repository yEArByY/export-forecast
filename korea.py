#LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime, timedelta

data=pd.read_csv('korea.csv')
data.drop([num for num in range (0,144)],inplace=True)

data.columns=['month','export']
data.month.str.split(expand=True)
data[[ 'year','month']]=data.month.str.split(expand=True)
data.month = data.month.str.title()
data['time']=data['year']+' '+data['month']
data.drop(['month','year'],axis=1,inplace=True)
data['time']=data['time'].apply(lambda entry:datetime.strptime(entry,'%Y %b'))
#show(plt.plot(data['time'],data['export']))
#print(data)

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df,n_steps):
    df=dc(df)
    df['time'] = pd.to_datetime(df['time'])

    df.set_index('time',inplace=True)

    for i in range(1,n_steps+1):
        df[f'export(t-{i})'] = df['export'].shift(i)

    df.dropna(inplace=True)
        
    return df

lookback=7
shifted_df=prepare_dataframe_for_lstm(data,lookback)
#print(shifted_df)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

#print(shifted_df_as_np)

X = shifted_df_as_np[:,1:]
Y = shifted_df_as_np[:,0]
print(X.shape)
print(Y.shape)
X= dc(np.flip(X, axis=1))

split_index=int(len(X)*0.6)

X_train=X[:split_index]
X_test=X[split_index:]

Y_train=Y[:split_index]
Y_test=Y[split_index:]

#X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

X_train=X_train.reshape((-1,lookback,1))
X_test=X_test.reshape((-1,lookback,1))

Y_train=Y_train.reshape((-1,1))
Y_test=Y_test.reshape((-1,1))

X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

X_train=torch.tensor(X_train).float()
X_test=torch.tensor(X_test).float()

Y_train=torch.tensor(Y_train).float()
Y_test=torch.tensor(Y_test).float()

X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i],self.Y[i]

train_dataset=TimeSeriesDataset(X_train,Y_train)
test_dataset=TimeSeriesDataset(X_test,Y_test)

from torch.utils.data import DataLoader

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


for p,batch in enumerate(train_loader):
    x_batch,y_batch=batch[0].to(device),batch[1].to(device)
    print(x_batch.shape,y_batch.shape)
    break


class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_stacked_layers=num_stacked_layers
        self.lstm=nn.LSTM(input_size,hidden_size,num_stacked_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,1)
        
    def forward(self,x):
        batch_size=x.size(0)
        h0=torch.zeros(self.num_stacked_layers,batch_size,self.hidden_size).to(device)
        c0=torch.zeros(self.num_stacked_layers,batch_size,self.hidden_size).to(device)
        out,p=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out

model=LSTM(1, 4, 1)
model.to(device)


def train_one_epoch():
    model.train(True)
    print(f'Epoch:{epoch+1}')
    running_loss=0.0
    
    for batch_index,batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device),batch[1].to(device)
        
        output = model(x_batch)
        loss=loss_function(output,y_batch)
        running_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_index%100 ==99:
            avg_loss_across_batches=running_loss/100
            print('Batch{0},Loss:{1:.3f}'.format(batch_index+1,avg_loss_across_batches))
            
            running_loss=0.0
    print()

def validate_one_epoch():
    model.train(False)
    running_loss=0.0
    
    for batch_index,batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device),batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss=loss_function(output,y_batch)
            running_loss+=loss
    
    avg_loss_across_batches=running_loss/len(test_loader)
    
    print('Val Loss:{0:.3f}'.format(avg_loss_across_batches))
    print(****************************************)
    print()

l_r=0.001
n_epoch=10
loss_function=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=l_r)

for epoch in range(n_epoch):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train,label='Actual Export')
plt.plot(y_train,label='Predicted Export')
plt.xlabel('Time')
plt.ylabel('Export')
plt.legent()
plt.show()

train_predictions = predicted.flatten()

dummies=np.zeros((X_train.shape[0],lookback+1))
dummies[:,0]=train_predictions
dummies = scalar.inverse_transform(dummies)

train_predictions = dc(dummies[:,0])
#train_predictions

dummies=np.zeros((X_train.shape[0],lookback+1))
dummies[:,0]= y_train.flatten()
dummies = scalar.inverse_transform(dummies)

new_y_train = dc(dummies[:,0])
#new_y_train

plt.plot(new_y_train,label='Actual Export')
plt.plot(train_predictions,label='Predicted Export')
plt.xlabel('Time')
plt.ylabel('Export')
plt.legend()
plt.show()


test_predictons=model(X_test.to(device)).detach().cpu.numpy().flatten()
dummies = np.zeros((X_test.shape[0],lookback+1))
dummies[:,0]=test_predictions
dummies = scalar.inverse_transform(dummies)

test_predictions = dc(dummies[;,0])
#test_predictions

dummies = np.zeros((X_test.shape[0],lookback+1))
dummies[:,0]=Y_test.flatten()
dummies = scalar.inverse_transform(dummies)

new_Y_test = dc(dummies[:,0])
#new_Y_test

plt.plot(new_Y_test,label='Actual Export')
plt.plot(train_predictions,label='Predicted Export')
plt.xlabel('Time')
plt.ylabel('Export')
plt.legend()
plt.show()

###################

#making predictions for the next 36 months
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
def create_dates(start,end):
    dates=[]
    while start <= end:
        dates.append(start)
        start += relativedelta(months=1)
    return dates

start_date=datetime(2024,6,1)
end_date=datetime(2027,6,1)
dates=create_dates(start_date,end_date)
print(create_dates(start_date,end_date))

for date in dates:
    last_row_data = shifted_df_as_np[-1,0:lookback]
    #print(last_row_data) 
    #last_row_data= dc(np.flip(last_row_data, axis=1))
    #new shape should be (1,3,1)
    last_row_data=last_row_data.reshape((-1,lookback,1))
    #print(last_row_data)
    last_row_data=torch.tensor(last_row_data).float()
    
    New_pred=model(last_row_data.to(device)).detach().cpu().numpy().flatten()#scaled new pred
    #print(last_row_data)
    #print(New_pred)#this is scaled
    
    
    #scaled_time=df1_np[0][0]
    
    # initialize list of lists
    #d1 = [[New_pred], [shifted_df_as_np[-1,0] ], [ shifted_df_as_np[-1,1]],[shifted_df_as_np[-1,2]]]
    #df1 = pd.DataFrame(d1, columns=['export','export(t-1)','export(t-2)','export(t-3)'])
    
    d1 = [{'export':New_pred,'export(t-1)':shifted_df_as_np[-1,0],'export(t-2)':shifted_df_as_np[-1,1],'export(t-3)':shifted_df_as_np[-1,2],'export(t-4)':shifted_df_as_np[-1,3],'export(t-5)':shifted_df_as_np[-1,4],'export(t-6)':shifted_df_as_np[-1,5],'export(t-7)':shifted_df_as_np[-1,6],'export(t-8)':shifted_df_as_np[-1,7],'export(t-9)':shifted_df_as_np[-1,8],'export(t-10)':shifted_df_as_np[-1,9],'export(t-11)':shifted_df_as_np[-1,10],'export(t-12)':shifted_df_as_np[-1,11]}]
    df1 = pd.DataFrame(d1)
    df1_inverted = scaler.inverse_transform(df1)
    
    d2= [{'time': date, 'export': df1_inverted[0][0]}]
    df2 = pd.DataFrame(d2)
    #df2 = pd.DataFrame(d2, columns=['time', 'export'])
    
    data = data._append(df2, ignore_index = True)
    #print(df2)
    #print(data)

        #add date and predicted export to csv data frame
    #data.loc[len(data.index)] = [date,int(New_pred)] 
    
    #generate np array from new dataframe(in data, it's export numbers, but output is scaled)
    shifted_df_as_np=prepare_dataframe_for_lstm(data,lookback).to_numpy()
    shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)
    #print(shifted_df_as_np)
print(data)
data.plot(x=data['time'],y=data['export'])




