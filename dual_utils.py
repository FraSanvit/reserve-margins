import pickle
import pandas as pd
import re

def duals_to_pickle(duals, path_to_duals):
    with open(path_to_duals, 'wb') as handle:
        pickle.dump(duals, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Saving dual variables as a .pickle file')
    
def load_duals(path_to_duals):
    
    with open(path_to_duals, 'rb') as handle:
        b = pickle.load(handle)
    
    print('Loading dual variables as a dictionary')
        
    return b

def process_system_balance_duals(system_balance_duals):
    column=system_balance_duals[0]

    info = column.str.split("[(::)]")
    info = pd.DataFrame(info.tolist(), index= info.index)
    info.drop([0,2,5],axis=1,inplace=True)
    
    car_and_time = info[3].str.split(",")
    info[['car','time']] = pd.DataFrame(car_and_time.tolist(), index= info.index)
    info.drop([3,4],axis=1,inplace=True)
    info.columns = ['region','carrier','timestep']
    info.region = pd.DataFrame(info.region.str.split("[']").to_list(), index=info.index)[1]
    info.carrier = pd.DataFrame(info.carrier.str.split("[']").to_list(), index=info.index)[0]
    info.timestep = pd.DataFrame(info.timestep.str.split("[']").to_list(), index=info.index)[1]
    info.timestep = info.timestep + ':00:00'
    info.timestep = pd.to_datetime(info.timestep)
    info['dual-value'] = system_balance_duals[1]

    return (info)

