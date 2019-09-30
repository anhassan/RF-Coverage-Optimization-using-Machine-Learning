import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import os, sys
import sqlite3
import sys
import re, sys, string
import csv
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import scipy.interpolate
import gmplot
import gmaps
import gmaps.datasets
import re, sys, string
import random
gmaps.configure(api_key="AIzaSyBk8CWaNZu_Ek6Vu-Qz-skDeHdxzwcnvQ8")
import sqlite3
import csv
import webbrowser , os
import scipy.interpolate
import gmplot
import seaborn as sns
import gmaps
import gmaps.datasets
import pandas as pd
import mgrs
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import scipy.interpolate

#=========================== Initiallizations========================================================

df_1 = pd.DataFrame()
df_2 = pd.DataFrame()
x_train = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
ch_df=pd.DataFrame()
n_ch_df=pd.DataFrame()



dict={"Zong":"4","Jazz":"MS2","Ufone":"MS3"," Telenor":"MS4"}
conn = sqlite3.connect('test_drive.db')

#================================ Main Functions======================================================================

def donothing():
    filewin = Toplevel(root)
    button = Button(filewin, text="Do nothing button")
    button.pack()

def add_quotes(word):
    word="'"+word+"'"
    return word

def outlier_range(data):
    print(len(data))
    print(sorted(data))
    if(len(data)%2==0):
        data=sorted(data)
        low_quart=np.median(data[0:int((len(data)/2))])
        up_quart=np.median(data[int((len(data)/2)):int(len(data))])
        outliers=[low_quart,up_quart]
    if(len(data)%2==1):
        data=sorted(data)
        low_quart=np.median(data[0:int(round(len(data)/2))])
        up_quart=np.median(data[int(round(len(data)/2)+1):int(len(data))])
        outliers=[low_quart,up_quart]
    int_quart=up_quart-low_quart
    print('int_quart',int_quart)
    print('low_quart',low_quart)
    print('up_quart',up_quart)
    start=low_quart-(2.5)*int_quart
    end=up_quart+(2.5)*int_quart
    ranged=[start,end,low_quart,up_quart]
    return ranged

def left_edge_mgrs(lat,lon):
    m=mgrs.MGRS()
    points=str(m.toMGRS(lat,lon))
    var = points[2:-1]
    point_left = var[0:9] + "0" + var[10:-1] + "0"
    return point_left

 def left_edge_lat(code):

     m = mgrs.MGRS()
     code= list(code.values)
     print('code=',code)
     result_tup = map(lambda x: m.toLatLon(x.encode()),code)
     print('result_tup',list(result_tup))
     result_tup=list(result_tup)
     rt_arr=np.array(result_tup)
     left_lat_list=rt_arr[:,0]
     return left_lat_list
     result_tup = m.toLatLon(code.encode())
    return list(result_tup)

def algo_one(x_train,y_train,x_test):
   regr = linear_model.LinearRegression()
   regr.fit(x_train,y_train)
   y_pred_lr = regr.predict(x_test)
   return y_pred_lr


def left_lat(code):
    m = mgrs.MGRS()
    result_tup=m.toLatLon(code.encode())
    return result_tup[0]

def left_lon(code):
    m = mgrs.MGRS()
    result_tup=m.toLatLon(code.encode())
    return result_tup[1]


def fetch_data(path,query):
    conn = sqlite3.connect('test_drive.db')
    cursor = conn.execute(query)
    output = cursor.fetchall()
    print("done")
    return output

def google_map_plots(df,zoom_level=17):
    lat_list=list(df.iloc[:,0].values)
    lon_list=list(df.iloc[:,1].values)
    new_lat_list= map(lambda nll: float(nll),lat_list)
    new_lat_list=list(new_lat_list)
    new_lon_list=map(lambda nll: float(nll),lon_list)
    new_lon_list=list(new_lon_list)
    centre_x=new_lat_list[0]
    centre_y=new_lon_list[0]
    #print('lat=',new_lat_list)
    #print('lon=',new_lon_list)
    gmap = gmplot.GoogleMapPlotter(centre_x,centre_y,zoom_level)
    gmap.scatter(new_lat_list,new_lon_list, '#4267b2', size=5, marker=False)
    gmap.draw("gmap.html")
    webbrowser.open('file://' + os.path.realpath("gmap.html"))

def google_covg_plots(df1,df2,zoom_level=17):
    lat_list_1=list(df1.iloc[:,0].values)
    lon_list_1=list(df1.iloc[:,1].values)
    new_lat_list_1= map(lambda nll: float(nll),lat_list_1)
    new_lat_list_1=list(new_lat_list_1)
    new_lon_list_1=map(lambda nll: float(nll),lon_list_1)
    new_lon_list_1=list(new_lon_list_1)
    lat_list_2 = list(df2.iloc[:, 0].values)
    lon_list_2 = list(df2.iloc[:, 1].values)
    new_lat_list_2 = map(lambda nll: float(nll), lat_list_2)
    new_lat_list_2 = list(new_lat_list_2)
    new_lon_list_2 = map(lambda nll: float(nll), lon_list_2)
    new_lon_list_2 = list(new_lon_list_2)
    centre_x=new_lat_list_1[0]
    centre_y=new_lon_list_1[0]
    #print('lat=',new_lat_list)
    #print('lon=',new_lon_list)
    gmap = gmplot.GoogleMapPlotter(centre_x,centre_y,zoom_level)
    gmap.scatter(new_lat_list_1,new_lon_list_1, '#4267b2', size=5, marker=False)
    gmap.scatter(new_lat_list_2, new_lon_list_2, '#830000', size=5, marker=False)
    gmap.draw("gmap.html")
    webbrowser.open('file://' + os.path.realpath("gmap.html"))


def spatial_binning(output):
    df = pd.DataFrame(output)
    df.columns = ['Lat', 'Lon', 'MS', 'RSRP']
    # extending the table to include Left_MGRS, Left_Lat and Left_Lon by using lambda functions
    df['Left_MGRS'] = df.apply(lambda x: left_edge_mgrs(x['Lat'], x['Lon']), axis=1)
    df['Left_Lat'] = df.apply(lambda x: left_lat(x['Left_MGRS']), axis=1)
    df['Left_Lon'] = df.apply(lambda x: left_lon(x['Left_MGRS']), axis=1)
    conn = sqlite3.connect('test_drive.db')
    # shifting this new table into the db for binning
    df.to_sql(con=conn, name='extended_table', if_exists='replace')
    #binning happening here (Yeyyy!!!)
    query = "SELECT Left_Lat,Left_Lon,MS,Avg(RSRP),Left_MGRS FROM extended_table GROUP BY Left_MGRS"
    out = fetch_data("", query)
    dft = pd.DataFrame(out)
    dft.columns = ['Lat', 'Lon', 'MS', 'RSRP','Left_MGRS']
    dft.drop(dft.index[0],inplace=True)
    if(p_1=="RSRP" or p_1=="RSCP"):
        dft = dft[dft['RSRP'] < -40.0]



    dft.to_sql(con=conn, name='extended_table', if_exists='replace')

    return dft




def setting_up_train_test_set(connection):
    query="SELECT lums_specs.Left_Lat_Lums,lums_specs.Left_Long_Lums, lums_specs.Left_edge_MGRS_LUMS,extended_table.RSRP FROM lums_specs LEFT JOIN extended_table ON lums_specs.Left_edge_MGRS_LUMS = extended_table.Left_MGRS"
    print(query)
    cursor=connection.execute(query)
    out=cursor.fetchall()
    df=pd.DataFrame(out)
    df.columns=['Lat','Lon','MGRS','RSRP']
    df.drop('MGRS',axis=1)
    df.to_sql(con=connection, name='combined_table', if_exists='replace')
    return df

def get_train_set(connection):
    query="SELECT Lat,Lon,RSRP FROM combined_table WHERE RSRP IS NOT NULL"
    cursor=connection.execute(query)
    out=cursor.fetchall()
    df=pd.DataFrame(out)
    df.columns=['Lat','Lon','RSRP']
    return df
def get_test_set(connection):
    query = "SELECT Lat,Lon,RSRP FROM combined_table WHERE RSRP IS NULL"
    cursor = connection.execute(query)
    out = cursor.fetchall()
    df = pd.DataFrame(out)
    df.columns = ['Lat', 'Lon','RSRP']
    df.drop('RSRP',axis=1,inplace=True)
    return df
def coverage_holes(x_train,y_train,x_test,y_pred):
    y_pred.columns=['RSRP']
    combined_val = pd.merge(y_train[['RSRP']], y_pred, how='outer')
    combined_x= pd.merge(x_train,x_test,how='outer')
    range = outlier_range(combined_val.values)
    df_full = pd.concat([combined_x['Lat'],combined_x['Lon'], combined_val['RSRP']], axis=1)
    coverage_holes = df_full[combined_val['RSRP'] <= -87.5]
    no_coverage_holes= df_full[combined_val['RSRP']>-87.5]
    print("coverage analysis",df_full.shape)
    return (coverage_holes,no_coverage_holes)

def net_map_ecno(x_train,y_train,x_test,y_pred):
    y_pred.columns = ['RSRP']
    combined_val = pd.merge(y_train[['RSRP']], y_pred, how='outer')
    combined_x = pd.merge(x_train, x_test, how='outer')
    df_full = pd.concat([combined_x['Lat'], combined_x['Lon'], combined_val['RSRP']], axis=1)
    print(df_full.head())


    range_1 = df_full[np.logical_and(df_full['RSRP'] >= -5, df_full['RSRP'] <= 0)]
    print("range1")
    print(range_1.head())
    print(range_1.shape)
    range_2 = df_full[np.logical_and(df_full['RSRP'] >= -9, df_full['RSRP'] <= -5)]
    print("range2")
    print(range_2.head())
    print(range_2.shape)
    range_3 = df_full[np.logical_and(df_full['RSRP'] >= -13, df_full['RSRP'] <= -9)]
    print("range3")
    print(range_3.head())
    print(range_3.shape)
    range_4 = df_full[np.logical_and(df_full['RSRP'] >= -16, df_full['RSRP'] <= -13)]
    print("range4")
    print(range_4.head())
    print(range_4.shape)
    range_5 = df_full[np.logical_and(df_full['RSRP'] >= -24, df_full['RSRP'] <= -16)]
    print("range5")
    print(range_5.head())
    print(range_5.shape)
    lat_list_1 = list(range_1.iloc[:, 0].values)
    lon_list_1 = list(range_1.iloc[:, 1].values)
    new_lat_list_1 = map(lambda nll: float(nll), lat_list_1)
    new_lat_list_1 = list(new_lat_list_1)
    new_lon_list_1 = map(lambda nll: float(nll), lon_list_1)
    new_lon_list_1 = list(new_lon_list_1)
    lat_list_2 = list(range_2.iloc[:, 0].values)
    lon_list_2 = list(range_2.iloc[:, 1].values)
    new_lat_list_2 = map(lambda nll: float(nll), lat_list_2)
    new_lat_list_2 = list(new_lat_list_2)
    new_lon_list_2 = map(lambda nll: float(nll), lon_list_2)
    new_lon_list_2 = list(new_lon_list_2)
    lat_list_3 = list(range_3.iloc[:, 0].values)
    lon_list_3 = list(range_3.iloc[:, 1].values)
    new_lat_list_3 = map(lambda nll: float(nll), lat_list_3)
    new_lat_list_3 = list(new_lat_list_3)
    new_lon_list_3 = map(lambda nll: float(nll), lon_list_3)
    new_lon_list_3 = list(new_lon_list_3)
    lat_list_4 = list(range_4.iloc[:, 0].values)
    lon_list_4 = list(range_4.iloc[:, 1].values)
    new_lat_list_4 = map(lambda nll: float(nll), lat_list_4)
    new_lat_list_4 = list(new_lat_list_2)
    new_lon_list_4 = map(lambda nll: float(nll), lon_list_4)
    new_lon_list_4 = list(new_lon_list_4)
    lat_list_5 = list(range_5.iloc[:, 0].values)
    lon_list_5 = list(range_5.iloc[:, 1].values)
    new_lat_list_5 = map(lambda nll: float(nll), lat_list_5)
    new_lat_list_5 = list(new_lat_list_5)
    new_lon_list_5 = map(lambda nll: float(nll), lon_list_5)
    new_lon_list_5 = list(new_lon_list_5)
    centre_x = new_lat_list_1[0]
    centre_y = new_lon_list_1[0]
    # print('lat=',new_lat_list)
    # print('lon=',new_lon_list)
    gmap = gmplot.GoogleMapPlotter(centre_x, centre_y, 17)
    gmap.scatter(new_lat_list_1, new_lon_list_1, '#B8FF33', size=5, marker=False)
    gmap.scatter(new_lat_list_2, new_lon_list_2, '#FFC300', size=5, marker=False)
    gmap.scatter(new_lat_list_3, new_lon_list_3, '#E67E22', size=5, marker=False)
    gmap.scatter(new_lat_list_4, new_lon_list_4, '#C0392B', size=5, marker=False)
    gmap.scatter(new_lat_list_5, new_lon_list_5, '#17202A', size=5, marker=False)
    gmap.draw("gmap.html")
    webbrowser.open('file://' + os.path.realpath("gmap.html"))

def net_map_snr(x_train,y_train,x_test,y_pred):
    y_pred.columns = ['RSRP']
    combined_val = pd.merge(y_train[['RSRP']], y_pred, how='outer')
    combined_x = pd.merge(x_train, x_test, how='outer')
    df_full = pd.concat([combined_x['Lat'], combined_x['Lon'], combined_val['RSRP']], axis=1)
    print(df_full.head())


    range_1 = df_full[df_full['RSRP'] >= 12.50]
    print("range1")
    print(range_1.head())
    print(range_1.shape)
    range_2 = df_full[np.logical_and(df_full['RSRP'] >= 10, df_full['RSRP'] <= 12.5)]
    print("range2")
    print(range_2.head())
    print(range_2.shape)
    range_3 = df_full[np.logical_and(df_full['RSRP'] >= 0, df_full['RSRP'] <= 10)]
    print("range3")
    print(range_3.head())
    print(range_3.shape)
    range_4 = df_full[df_full['RSRP'] <= 0]
    print("range4")
    print(range_4.head())
    print(range_4.shape)


    lat_list_1 = list(range_1.iloc[:, 0].values)
    lon_list_1 = list(range_1.iloc[:, 1].values)
    new_lat_list_1 = map(lambda nll: float(nll), lat_list_1)
    new_lat_list_1 = list(new_lat_list_1)
    new_lon_list_1 = map(lambda nll: float(nll), lon_list_1)
    new_lon_list_1 = list(new_lon_list_1)
    lat_list_2 = list(range_2.iloc[:, 0].values)
    lon_list_2 = list(range_2.iloc[:, 1].values)
    new_lat_list_2 = map(lambda nll: float(nll), lat_list_2)
    new_lat_list_2 = list(new_lat_list_2)
    new_lon_list_2 = map(lambda nll: float(nll), lon_list_2)
    new_lon_list_2 = list(new_lon_list_2)
    lat_list_3 = list(range_3.iloc[:, 0].values)
    lon_list_3 = list(range_3.iloc[:, 1].values)
    new_lat_list_3 = map(lambda nll: float(nll), lat_list_3)
    new_lat_list_3 = list(new_lat_list_3)
    new_lon_list_3 = map(lambda nll: float(nll), lon_list_3)
    new_lon_list_3 = list(new_lon_list_3)
    lat_list_4 = list(range_4.iloc[:, 0].values)
    lon_list_4 = list(range_4.iloc[:, 1].values)
    new_lat_list_4 = map(lambda nll: float(nll), lat_list_4)
    new_lat_list_4 = list(new_lat_list_2)
    new_lon_list_4 = map(lambda nll: float(nll), lon_list_4)
    new_lon_list_4 = list(new_lon_list_4)

    centre_x = new_lat_list_1[0]
    centre_y = new_lon_list_1[0]
    # print('lat=',new_lat_list)
    # print('lon=',new_lon_list)
    gmap = gmplot.GoogleMapPlotter(centre_x, centre_y, 17)
    gmap.scatter(new_lat_list_1, new_lon_list_1, '#B8FF33', size=5, marker=False)
    gmap.scatter(new_lat_list_2, new_lon_list_2, '#FFC300', size=5, marker=False)
    gmap.scatter(new_lat_list_3, new_lon_list_3, '#E67E22', size=5, marker=False)
    gmap.scatter(new_lat_list_4, new_lon_list_4, '#C0392B', size=5, marker=False)
    gmap.draw("gmap.html")
    webbrowser.open('file://' + os.path.realpath("gmap.html"))

def net_map_rsrp(x_train,y_train,x_test,y_pred):
    y_pred.columns = ['RSRP']
    combined_val = pd.merge(y_train[['RSRP']], y_pred, how='outer')
    combined_x = pd.merge(x_train, x_test, how='outer')
    df_full = pd.concat([combined_x['Lat'], combined_x['Lon'], combined_val['RSRP']], axis=1)
    print(df_full.head())
    df_full.to_sql(con=conn, name='checking_problem', if_exists='replace')
    print("heat map analysis", df_full.shape)
    range_1 = df_full[np.logical_and(df_full['RSRP'] >= -70.0, df_full['RSRP'] <= -30.0)]
    print("range1")
    print(range_1.head())
    print(range_1.shape)
    range_2 = df_full[np.logical_and(df_full['RSRP'] >= -74, df_full['RSRP'] < -70)]
    print("range2")
    print(range_2.head())
    print(range_2.shape)
    range_3 = df_full[np.logical_and(df_full['RSRP'] >= -83, df_full['RSRP'] < -74)]
    print("range3")
    print(range_3.head())
    print(range_3.shape)
    range_4 = df_full[np.logical_and(df_full['RSRP'] >= -95, df_full['RSRP'] < -83)]
    print("range4")
    print(range_4.head())
    print(range_4.shape)
    range_5 = df_full[np.logical_and(df_full['RSRP'] >= -140, df_full['RSRP'] < -95)]
    print("range5")
    print(range_5.head())
    print(range_5.shape)
    lat_list_1 = list(range_1.iloc[:, 0].values)
    lon_list_1 = list(range_1.iloc[:, 1].values)
    new_lat_list_1 = map(lambda nll: float(nll), lat_list_1)
    new_lat_list_1 = list(new_lat_list_1)
    new_lon_list_1 = map(lambda nll: float(nll), lon_list_1)
    new_lon_list_1 = list(new_lon_list_1)
    lat_list_2 = list(range_2.iloc[:, 0].values)
    lon_list_2 = list(range_2.iloc[:, 1].values)
    new_lat_list_2 = map(lambda nll: float(nll), lat_list_2)
    new_lat_list_2 = list(new_lat_list_2)
    new_lon_list_2 = map(lambda nll: float(nll), lon_list_2)
    new_lon_list_2 = list(new_lon_list_2)
    lat_list_3 = list(range_3.iloc[:, 0].values)
    lon_list_3 = list(range_3.iloc[:, 1].values)
    new_lat_list_3 = map(lambda nll: float(nll), lat_list_3)
    new_lat_list_3 = list(new_lat_list_3)
    new_lon_list_3 = map(lambda nll: float(nll), lon_list_3)
    new_lon_list_3 = list(new_lon_list_3)
    lat_list_4 = list(range_4.iloc[:, 0].values)
    lon_list_4 = list(range_4.iloc[:, 1].values)
    new_lat_list_4 = map(lambda nll: float(nll), lat_list_4)
    new_lat_list_4 = list(new_lat_list_2)
    new_lon_list_4 = map(lambda nll: float(nll), lon_list_4)
    new_lon_list_4 = list(new_lon_list_4)
    lat_list_5 = list(range_5.iloc[:, 0].values)
    lon_list_5 = list(range_5.iloc[:, 1].values)
    new_lat_list_5 = map(lambda nll: float(nll), lat_list_5)
    new_lat_list_5 = list(new_lat_list_5)
    new_lon_list_5 = map(lambda nll: float(nll), lon_list_5)
    new_lon_list_5 = list(new_lon_list_5)
    centre_x = new_lat_list_1[0]
    centre_y = new_lon_list_1[0]
    # print('lat=',new_lat_list)
    # print('lon=',new_lon_list)
    gmap = gmplot.GoogleMapPlotter(centre_x, centre_y, 17)
    gmap.scatter(new_lat_list_1, new_lon_list_1, '#B8FF33', size=5, marker=False)
    gmap.scatter(new_lat_list_2, new_lon_list_2, '#FFC300', size=5, marker=False)
    gmap.scatter(new_lat_list_3, new_lon_list_3, '#E67E22', size=5, marker=False)
    gmap.scatter(new_lat_list_4, new_lon_list_4, '#C0392B', size=5, marker=False)
    gmap.scatter(new_lat_list_5, new_lon_list_5, '#17202A', size=5, marker=False)
    gmap.draw("gmap.html")
    webbrowser.open('file://' + os.path.realpath("gmap.html"))

def export_raw_data(df):
    print("here boys!!!")
    df.to_csv("raw_data.csv",encoding='utf-8', index=False)
def export_covg_holes(df):
    df.to_csv("coverage_holes.csv",encoding='utf-8', index=False)
def export_no_covg_holes(df):
    df.to_csv("good_coverage.csv",encoding='utf-8', index=False)
def export_combined_data(df):
    df.to_csv("combined_data.csv",encoding='utf-8', index=False)


def algo_two(x_train,y_train,x_test,d=5):
    model = make_pipeline(PolynomialFeatures(d), Ridge())
    model.fit(x_train,y_train)
    y_pred_pr = model.predict(x_test)
    return y_pred_pr

def algo_three(x_train,y_train,x_test,k='rbf'):
    svr_poly=SVR(kernel=k,degree=5,C=1e3)
    svr_poly.fit(x_train,y_train)
    y_pred_svr=svr_poly.predict(x_test)
    return y_pred_svr

#============================== Button Functions===========================================================
def google_scatter_plot():
    google_map_plots(df_1)
def google_scatter_plot_binning():
    google_map_plots(df_2)
def coverage_holes_draw():
    google_covg_plots(n_ch_df,ch_df)
def net_map_draw():
    net_map_rsrp(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)
def histogram_plot():
    print("inside_hist")
    print(pred_df.shape)
    plt.title(variable1.get())
    plt.figure()
    plt.ylabel('counts')
    plt.xlabel('RSRP')
    print("inside_hist",pred_df['RSRP'])
    plt.hist(pred_df['RSRP'])
    plt.show()
def box_plot():
    sns.set()
    plt.figure()
    plt.title(variable1.get())
    sns.boxplot(y=pred_df['RSRP'])
    plt.show()
def bee_swarm_plot():
    sns.set()
    plt.figure()
    plt.title(variable1.get())
    sns.swarmplot(y=pred_df['RSRP'])
    plt.show()
def plot_live_data():
    import time
    arr=pred_df['RSRP'].values.reshape(-1,1)
    xdata=[]
    ydata=[]
    plt.show()
    axes=plt.gca()
    axes.set_xlim(0,len(arr))
    axes.set_ylim(min(arr),max(arr))
    line,=axes.plot(xdata,ydata,'b-')
    for i in range(len(arr)):
        xdata.append(i)
        ydata.append(arr[i])
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)
        time.sleep(0.1)
plt.show()

def initiallizer():
    query=""
    print("p_1=",p_1)
    print("t_1=",t_1)
    print("algos",algos)
    #query="SELECT Latitude,Longitude,MS,RSRP FROM mixed_data WHERE MS="+add_quotes(dict[variable1.get()])+" AND Time BETWEEN "+add_quotes(usertext_start.get() + " "+usertext_time1.get())+" AND "+ add_quotes(usertext_end.get()+" "+usertext_time2.get())
    if(p_1=="RSRP" and t_1=="Loading"):
        query = "SELECT Longitude,Latitude,MS,RSRP FROM final_mixed_table WHERE MS=" + add_quotes(dict[variable1.get()])
    if(p_1=="ECNO" and t_1=="Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM ECNO WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "RSCP" and t_1 == "Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM RSCP WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "SNR" and t_1 == "Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM SNR WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "RSRP" and t_1 == "Non-Loading"):
        query = "SELECT Longitude,Latitude,MS,RSRP FROM final_mixed_table_NL WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "ECNO" and t_1 == "Non-Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM ECNO_NL WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "RSCP" and t_1 == "Non-Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM RSCP_NL WHERE MS=" + add_quotes(dict[variable1.get()])
    if (p_1 == "SNR" and t_1 == "Non-Loading"):
        query = "SELECT Latitude,Longitude,MS,RSRP FROM SNR_NL WHERE MS=" + add_quotes(dict[variable1.get()])

    print('query= ',query)
    output = fetch_data("", query)
    global df_1,df_2
    df_1 = pd.DataFrame(output)
    print(df_1.head())
    df_2 = spatial_binning(output)
    print(df_2.head())
    #google_map_plots(df_1)
    global df_combined,df_train,df_test
    df_combined = setting_up_train_test_set(conn)
    df_train = get_train_set(conn)
    df_test = get_test_set(conn)
    #google_map_plots(df_2)
    global x_train,y_train,x_test
    x_train = df_train[['Lat', 'Lon']].values
    y_train = df_train[['RSRP']].values
    x_test = df_test[['Lat', 'Lon']].values
    if algos=='Linear Regression':
        pred = algo_one(x_train, y_train, x_test)
    if algos=='Polynomial Regression':

        pred = algo_two(x_train, y_train, x_test)
    if algos=='Support Vector Regression':

        pred = algo_three(x_train, y_train, x_test)

    global pred_df,ch_df,n_ch_df
    pred_df = pd.DataFrame(pred)
    print("printing predicted values")
    print(pred_df.head())
    
    (ch, n_ch) = coverage_holes(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)
    
    ch_df=pd.DataFrame(ch)
    n_ch_df = pd.DataFrame(n_ch)
    

    if variable4.get()=="Network Before Binning":
        google_map_plots(df_1)
    if variable4.get()=="Network After Binning":
        google_map_plots(df_2)
    if variable4.get()=="Coverage Holes":
        google_covg_plots(n_ch_df, ch_df)
    if variable4.get()=="Network Heat Map" and p_1=="RSRP":
        net_map_rsrp(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)
    if variable4.get()=="Network Heat Map" and p_1=="ECNO":
        net_map_ecno(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)
    if variable4.get()=="Network Heat Map" and p_1=="RSCP":
        net_map_rsrp(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)
    if variable4.get()=="Network Heat Map" and p_1=="SNR":
        net_map_snr(df_train[['Lat', 'Lon']], df_train[['RSRP']], df_test[['Lat', 'Lon']], pred_df)

    if exp=='Coverage Holes':
        export_covg_holes(ch)
    elif exp == ' No Coverage Holes':
        export_no_covg_holes(n_ch)
    elif exp == 'Raw Data':
        
        export_raw_data(df_combined)
    elif exp == 'Interpolated Data':
        export_combined_data(pred_df)

    f = Figure(figsize=(3, 2), dpi=100)
    a = f.add_subplot(111)
    # a.set_ylabel("Histogram")
    # a.hist([100, 200, 300, 400, 500, 600, 700, 800])

    a.hist(pred_df['RSRP'])
    a.set_ylabel("Histogram")

    canvas = FigureCanvasTkAgg(f, root)
    # canvas.show()
    # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # toolbar = NavigationToolbar2TkAgg(canvas, root)
    # toolbar.update()
    canvas._tkcanvas.pack(side=TOP, fill=X, expand=0.5)

    if variable5.get()=="Network Health Histogram":
        
        histogram_plot()
    if variable5.get()=="Box Plot":
        box_plot()
    if variable5.get()=="Bee Swarm Plot":
        bee_swarm_plot()
    print("about to be done!!!!")









   

root = tk.Tk()
# root.resizable(width=FALSE, height=FALSE)
root.title("Network Health Monitoring and Estimation Suite")
root.geometry("1000x600")
root.iconbitmap('LUMS.ico')

# img = Image.open("LUMS.gif")
img = Image.open("LUMS.png")
img = img.resize((150,115))
img = ImageTk.PhotoImage(img)

img1 = Image.open("ICT_logo.png")
# img1 = img1.resize((150, 41))
img1 = ImageTk.PhotoImage(img1)

# img = ImageTk.PhotoImage(Image.open("LUMS.gif"))
Algorithm = IntVar()

variable1 = StringVar()
variable2 = StringVar()
variable3 = StringVar()
variable4=StringVar()
variable5=StringVar()
variable6=StringVar()
variable7=StringVar()
variable8=StringVar()

algos=StringVar()


exp=StringVar()


t_1=StringVar()


p_1=StringVar()


arfcn = 0
route = ""
time = ""
threshold = 0
network = ""







# ---------------------GUI---------------------------------
first = Frame(root, width=1000, height=200)
first.pack(side=TOP)

middle = Frame(root, width=1000, height=30)
middle.pack()

second = Frame(root, width=1000, height=410)
second.pack()

# third = Frame(root, width=1000, height=100)
# third.pack(side=LEFT)


# panel.pack(side = "bottom", fill = "both", expand = "yes")

lums_logo = Label(first, image = img)
lums_logo.grid(row=0, column=0)

label_title = Label(first, font=('Verdena', 17, 'bold'), text=" Network Health Monitoring and Estimation Suite ", fg="Dark Blue", bd=20,
                    anchor='center')
label_title.grid(row=0, column=1)

ict_logo = Label(first, image = img1)
ict_logo.grid(row=0, column=2)


# --------------------------NETWORK SELECT MENU-----------------------------------------#
label_3 = Label(second, text="Select Operator:        ", fg="Dark Blue", font=('arial', 11, 'bold'))
label_3.grid(row=3, column=0)

NETWORKS = ["Ufone","Jazz","Zong","Telenor"] #etc

# variable1 = StringVar(second)
variable1.set(NETWORKS[0]) # default value

w = OptionMenu(second, variable1, *NETWORKS)  #variable1 = network selected
w.config(padx=12, pady=6, bd=16, fg="black", font=('arial', 10, 'bold'), width=22, bg="silver")
w.grid(row=3,column=1)

spacey = Label(second, text="")
spacey.grid(row=4, column=2)


# --------------------------ALGORTHIM SELECT MENU-----------------------------------------#
label_3 = Label(second, text="Select Technology:        ", fg="Dark Blue", font=('arial', 11, 'bold'))
label_3.grid(row=5, column=0)

TECH= ["2G", "3G", "4G"]  # etc


variable2.set(TECH[1])  # default value

w = OptionMenu(second, variable2, *TECH)
w.config(padx=12, pady=6, bd=16, fg="black", font=('arial', 10, 'bold'), width=22, bg="silver")
w.grid(row=5, column=1)

spacey = Label(second, text="")
spacey.grid(row=6, column=2)


label_2 = Label(second, text="Select the type of Geographical Map:           ", fg="Dark Blue", font=('arial', 11, 'bold'))
label_2.grid(row=7, column=0)




MAPS= ["Network Before Binning", "Network After Binning","Coverage Holes","Network Heat Map"]  # etc


variable4.set(MAPS[0])  # default value

w = OptionMenu(second, variable4, *MAPS)
w.config(padx=12, pady=6, bd=16, fg="black", font=('arial', 10, 'bold'), width=22, bg="silver")
w.grid(row=7, column=1)



spacex = Label(second, text="                ")
spacex.grid(row=8, column=2)






label_2 = Label(second, text="Select the type of Statistical Plot:           ", fg="Dark Blue", font=('arial', 11, 'bold'))
label_2.grid(row=9, column=0)

STATS= ["Network Health Histogram", "Box Plot","Bee Swarm Plot"]  # etc


variable5.set(STATS[0])  # default value

w = OptionMenu(second, variable5, *STATS)
w.config(padx=12, pady=6, bd=16, fg="black", font=('arial', 10, 'bold'), width=22, bg="silver")
w.grid(row=9, column=1)


spacex = Label(second, text="                ")
spacex.grid(row=10, column=2)


button_12 = Button(second, padx=12, pady=6, bd=16, fg="black", font=('arial', 10, 'bold'), width=22, bg="silver",
                  text="Run", command=lambda :initiallizer())              #Replace msg with the function of coverage holes
button_12.grid(row=11, column=1)






# --------------------------MENU BAR-----------------------------------------#
menubar = Menu(root)

filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=donothing)
filemenu.add_command(label="Open", command=donothing)
filemenu.add_command(label="Save", command=donothing)
filemenu.add_command(label="Save as...", command=donothing)
filemenu.add_command(label="Close", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="Undo", command=donothing)
editmenu.add_separator()
editmenu.add_command(label="Cut", command=donothing)
editmenu.add_command(label="Copy", command=donothing)
editmenu.add_command(label="Paste", command=donothing)
editmenu.add_command(label="Delete", command=donothing)
editmenu.add_command(label="Select All", command=donothing)
menubar.add_cascade(label="Edit", menu=editmenu)

def p1():
    global p_1
    p_1="RSS1"

def p2():
    global p_1
    p_1="BER"

def p3():
    global p_1
    p_1="RSRP"
    print("here")

def p4():
    global p_1
    p_1="ECNO"

def p5():
    global p_1
    p_1="RSCP"

def p6():
    global p_1
    p_1 = "SNR"


par = Menu(menubar, tearoff=0)
#par.add_separator()
par.add_command(label="2G Parameters", command=donothing)
par.add_separator()
par.add_checkbutton(label="RSSI", command=lambda: p1())
par.add_checkbutton(label="BER", command=lambda: p2())
par.add_separator()
par.add_command(label="3G Parameters", command=donothing)
par.add_separator()
par.add_checkbutton(label="RSRP", command=lambda: p3())
par.add_checkbutton(label="ECNO", command=lambda: p4())
par.add_separator()
par.add_command(label="4G Parameters", command=donothing)
par.add_separator()
par.add_checkbutton(label="RSCP", command=lambda: p5())
par.add_checkbutton(label="SNR", command=lambda: p6())
menubar.add_cascade(label="Select Parameter", menu=par)

def t1():
    global t_1
    t_1="Loading"
    print("here2")
def t2():
    global t_1
    t_1="Non-Loading"
timemenu = Menu(menubar, tearoff=0)
timemenu.add_checkbutton(label="Loading", command=lambda: t1())
timemenu.add_checkbutton(label="Non-Loading", command=lambda: t2())
menubar.add_cascade(label="Select Time", menu=timemenu)



def LR():
    global algos
    algos="Linear Regression"



def PR():
    global algos
    algos="Polynomial Regression"


def VR():

    global algos
    algos="Support Vector Regression"


algomenu = Menu(menubar, tearoff=0)
algomenu.add_checkbutton(label="Linear Regression", command=lambda: LR())
algomenu.add_checkbutton(label="Polynomial Regression", command= lambda: PR())
algomenu.add_checkbutton(label="Support Vector Regression", command=lambda: VR())
menubar.add_cascade(label="Select Algorithm", menu=algomenu)

def CH():
    global exp
    exp="Coverage Holes"


def NCH():

    global exp
    exp="No Coverage Holes"


def RD():

    global exp
    exp="Raw Data"


def ID():

    global exp
    print("go")
    exp="Interpolated Data"


exportmenu = Menu(menubar, tearoff=0,relief=RAISED )
exportmenu.add_checkbutton(label="Coverage Holes", command=lambda:CH())
exportmenu.add_checkbutton(label="No Coverage Holes", command=lambda: NCH())
exportmenu.add_checkbutton(label="Raw Data", command=lambda : RD())
exportmenu.add_checkbutton(label="Interpolated Data", command=lambda: ID())

menubar.add_cascade(label="Export Data", menu=exportmenu)


f = Figure(figsize=(3, 2), dpi=100)
a = f.add_subplot(111)
a.set_ylabel("Histogram")
a.hist([100, 200, 300, 400, 500, 600, 700, 800])





status = Label(root, text="Developed by LUMS AdCom Lab.       Version: 1.0a1 (alpha release 1)      "
                          "Contact @ support@lums.edu.pk", bd=1, relief=SUNKEN, anchor='e')
root.config(menu=menubar)
root.mainloop()



