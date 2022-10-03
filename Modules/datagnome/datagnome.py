import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import interact
import phik

from sklearn.linear_model import LinearRegression

import scipy
import scipy.fftpack

from sklearn.linear_model import LinearRegression


def ttt_split_length(X,y,test_size=0.25):
    """Test train split by test size
    
    Parameters
    ---------
    X:          X
    y:          y
    test_size:  Proportion if dataset to be allocated to the test sample
    """
    sl = len(X)-int(len(X)*test_size)
    X_train = X[0:sl]
    y_train = y[0:sl]
    X_test = X[sl:]
    y_test = y[sl:]
    return X_train,X_test,y_train,y_test

def ttt_split_dtime(X,y,test_start,dtcol=""):
    """Test train split by datetime value
    
    Parameters
    ---------
    X:           X
    y:           y
    dtcol:       
    test_start:  
    """
    if not dtcol:
        cond = X.index >= test_start
    else:
        cond = X[dtcol] >= test_start
    X_train = X.loc[~cond]
    y_train = y.loc[~cond]
    X_test = X.loc[cond]
    y_test = y.loc[cond]
    print(f"Test subset share of full dataset: {X_test.shape[0]/X.shape[0]}")
    return X_train,X_test,y_train,y_test

def df_flexplot(df):
    """Interactive plot
    
    Parameters
    ---------
    df:  Pandas dataframe
    """
    dfnum = df_allcatnum(df)
    dxw = widgets.IntSlider(min=3, max=15, step=1, value=6)
    dyw = widgets.IntSlider(min=3, max=15, step=1, value=5)
    pltypw = widgets.Dropdown(options=["kdeplot","histplot","catplot","barplot","countplot",
                                       "lineplot","scatterplot","lmplot","jointplot","boxplot"])
    colw = widgets.Dropdown(options=df.columns.to_list())
    colw2 = widgets.Dropdown(options=[None]+df.columns.to_list())
    subcolw = widgets.Dropdown(options=[None]+df.columns.to_list())
    huew = widgets.Dropdown(options=[None]+df.columns.to_list())
    kdew = widgets.Dropdown(options=[False,True])
    logsw = widgets.Dropdown(options=[None,True])
    cumuw = widgets.Dropdown(options=[False,True])
    estlw = widgets.Dropdown(options=["mean",None,"min","max","std","sum"])
    estl2w = widgets.Dropdown(options=[("mean", np.mean), ("std", np.std)])
    jokinw = widgets.Dropdown(options=["scatter","kde","hist"])
    kndcow = widgets.Dropdown(options=["strip","swarm","box","violin","boxen","point",
                                      "bar","count"])
    
    @interact(dx=dxw,dy=dyw,pt=pltypw,col=colw,col2=colw2,subcol=subcolw,hue=huew,
              logs=logsw,kde=kdew,cu=cumuw,estl=estlw,est2=estl2w,jok=jokinw,
              kind=kndcow)
    def f(dx,dy,pt,col,col2,subcol,hue,logs,kde,cu,estl,est2,jok,kind):
        kdew.layout.display = "none"
        logsw.layout.display = "none"
        cumuw.layout.display = "none"
        estlw.layout.display = "none"
        estl2w.layout.display = "none"
        jokinw.layout.display = "none"
        subcolw.layout.display = "none"
        kndcow.layout.display = "none"
        asp = dx/dy
        
        if pt not in ["catplot","lmplot","jointplot"]:
            plt.figure(figsize=(dx,dy))
        
        if pt == "kdeplot":
            logsw.layout.display = "block"
            cumuw.layout.display = "block"
            g = sns.kdeplot(data=dfnum,x=col,hue=hue,log_scale=logs,
                           cumulative=cu)
        elif pt == "histplot":
            kdew.layout.display = "block"
            logsw.layout.display = "block"
            cumuw.layout.display = "block"
            g = sns.histplot(data=df,x=col,hue=hue,log_scale=logs,kde=kde,
                            cumulative=cu)
        elif pt == "catplot":
            subcolw.layout.display = "block"
            kndcow.layout.display = "block"
            if kind in ["point","bar"]:
                estl2w.layout.display = "block"
            g = sns.catplot(data=df,x=col, y=None if kind == "count" else col2, 
                            hue=hue,col=subcol, kind=kind,
                            estimator=est2,height=dy, aspect=asp)
        elif pt == "countplot":
            g = sns.countplot(data=df,x=col,hue=hue)
        elif pt == "barplot":
            estl2w.layout.display = "block"
            g = sns.barplot(data=df,x=col,y=col2,hue=hue,estimator=est2)
        elif pt == "lineplot":
            estlw.layout.display = "block"
            g = sns.lineplot(data=df, x=col, y=col2,hue=hue,estimator=estl)
        elif pt == "scatterplot":
            g = sns.scatterplot(data=dfnum,x=col,y=col2,hue=hue)
        elif pt == "lmplot":
            subcolw.layout.display = "block"
            g = sns.lmplot(data=dfnum,x=col,y=col2,hue=hue,col=subcol,
                          height=dy, aspect=asp)
        elif pt == "jointplot":
            jokinw.layout.display = "block"
            g = sns.jointplot(data=dfnum, x=col, y=col2, hue=hue,kind=jok,
                             height=dx)
        elif pt == "boxplot":
            g = sns.boxplot(data=dfnum,x=col,y=col2,hue=hue)
        if hue != None and pt not in ["lmplot","catplot"]:
            g.legend_._set_loc(2)
            g.legend_.set_bbox_to_anchor((1.02, 1))
        plt.show()

def df_cobahiscplots(df,cols,kolm=4,ch=4,pw=16,pltyp="countplot",target=False,hue=None,logs=False,kde=False,est=False,stat="count"):
    """Countplot, barplot, histplot or scatterplot
    
    Parameters
    ---------
    kolm:   Number of plot grid columns
    ch:     Subplot height
    pw:     Total width of plot grid
    pltyp:  Type of plot {countplot,histplot,barplot,scatterplot}, default countplot
    target: Y-axis column (only barplots and scatterplots)
    hue:    Hue {Col name | False}, default False (not used in barplots)
    logs:   If true use logaritmic scaling {True | False}, default False (only histplots)
    kde:    If True, add compute a kernel density estimate: {True | False}, default False (only histplots)
    est:    Statistical function to estimate within each categorical bin. {False | np.mean | np.median | len | sum | max | min}, default False
    stat:   Aggregate statistic to compute in each bin: {count | frequency | probability | percent | density}, default count (only histplot)  
    
    Acknowledgement:
    https://www.kaggle.com/vascodegama/automated-seaborn-plot-functions
    """
    kolm = min(kolm,len(cols))
    rows = math.ceil(len(cols)/kolm)
    fig, axs = plt.subplots(rows, kolm, figsize=(pw,ch*rows))
    
    for i, ax in zip(cols,axs.flat if kolm*rows>1 else [axs]):
        # Histplots
        if pltyp == "histplot":
            sns.histplot(data=df, x=i, ax=ax, log_scale=logs, hue=hue, stat=stat, kde=kde)
            ax.set(xlabel="",ylabel=stat,title=i)
        # Countplots
        elif pltyp == "countplot":
            sns.countplot(data=df, x=i, ax=ax, hue=hue)
            ax.set(xlabel="",ylabel="count",title=i)
        # Barplots
        elif pltyp == "barplot":
            if est:
                sns.barplot(data=df, x=i, y=target, ax=ax, ci=None, estimator=est)
                ax.set(xlabel="",ylabel=f"{target} ({est.__name__})",title=i)
            else:
                sns.barplot(data=df, x=i, y=target, ax=ax, ci=None)
                ax.set(xlabel="",ylabel=f"{target} (mean)",title=i)
        
        # Scatterplots
        if (pltyp == "scatterplot") and target:
            sns.scatterplot(data=df, x=i, y=target, ax=ax)
            ax.set(xlabel="",ylabel=target,title=i)
        
        ax.set_title(i)
        
    # Remove any unused axs
    for i in axs.flat[::-1][:rows*kolm-len(cols)] if kolm*rows>1 else []:
        i.set_axis_off()
    plt.show()
    
def create_feature_lists(data):
    '''Create lists of all_numeric, continuous_numeric (>25 unique values), discrete_numeric, 
    and categorical features excluding high-cardinality features < 13
    
    Acknowledgement:
    https://www.kaggle.com/vascodegama/automated-seaborn-plot-functions
    '''
    catcol = data.select_dtypes(include=["bool","object","category"]).columns
    NUM_FEAT = [c for c in data.columns if c not in catcol]
    CONT_FEAT = [c for c in NUM_FEAT if len(data[c].unique()) > 25]
    DISC_FEAT = [c for c in NUM_FEAT if len(data[c].unique()) < 25]
    CAT_FEAT = [c for c in data.columns if c in catcol]
    return NUM_FEAT, CONT_FEAT, DISC_FEAT, CAT_FEAT

def df_metaplot(df,cols=False,hue=None,diffrel=False):
    """Overview plot for data cleaning  
    
    Parameters
    ---------
    df:       Pandas dataframe
    cols:     Target column if any
    diffrel:  {True, False}, default False
              True:  Use .pct_change() to calculate step change
              False: Use .diff() to calculate step change
    """
    if not cols:
        cols = df.columns
    rows = len(cols)
    kolm = 5
    catcol = df.select_dtypes(include=["bool","object","category"]).columns
    fig, axs = plt.subplots(rows+1, kolm, figsize=(20,5*rows))
    
    # Index column
    ix = [i for i in range(len(df.index))]
    sns.lineplot(data=df, x=ix, y=df.index,ax=axs[0][0],hue=hue)
    sns.histplot(data=df, x=df.index, ax=axs[0][1], stat="count")
    df_drwiqrlim(df.index.to_series(),axs[0][1])
    sns.lineplot(data=df, x=ix, y=df.index.to_series().diff(),ax=axs[0][2],hue=hue)
    axs[0][0].set_ylabel("Index")
    axs[0][2].set_ylabel("Delta-index")
    
    # Value columns
    for i,c in enumerate(cols):
        if c not in catcol and len(df[c].unique()) >= 1:
            ytmp = df[c].pct_change(periods=1) if diffrel else df[c].diff()
            sns.lineplot(data=df, x=df.index, y=c,ax=axs[i+1][0],hue=hue)
            sns.histplot(data=df, x=c, ax=axs[i+1][1], stat="count",hue=hue)
            sns.lineplot(data=df, x=df.index, y=ytmp,ax=axs[i+1][2],hue=hue)
            sns.histplot(data=df, x=ytmp, ax=axs[i+1][3], stat="count",hue=hue)
            axs[i+1][2].set_ylabel(f"Delta-{c}")
            axs[i+1][3].set_xlabel(f"Delta-{c}")
            
            # IQR
            df_drwiqrlim(ytmp,axs[i+1][3])
            df_drwiqrlim(df[c],axs[i+1][1])
            
            # FFT-plot
            N=df[c].shape[0]
            y=np.array(df[c])
            yf = np.abs(scipy.fftpack.fft(y))
            xf = scipy.fftpack.fftfreq(N, d=1)
            sns.lineplot(x=np.abs(xf[:xf.size//2]),y=np.abs(yf[:yf.size//2]), ax=axs[i+1][4])
        else:
            sns.countplot(data=df, x=c, ax=axs[i+1][1], hue=hue)
            axs[i+1][1].set(xlabel=c,ylabel="count",title="")
    plt.show()

def df_drwiqrlim(y,plt):
    """Helper function for df_metaplot"""
    q1 = y.quantile(.25)
    q3 = y.quantile(.75)
    dq = 1.5*(q3-q1)
    plt.axvline(q1-dq, 0, 0.95, color='red', linestyle='--')
    plt.axvline(q3+dq, 0, 0.95, color='red', linestyle='--')
    plt.text(q1-dq*0.95, plt.get_ylim()[1]*0.7, "Q1-1.5*IQR",rotation="vertical") #plt.ylim() for figure
    plt.text(q3+dq*1.05, plt.get_ylim()[1]*0.7, "Q3+1.5*IQR",rotation="vertical")

def df_corrbas(df,col=False,method="pearson"):
    """Correlation plot using methods built into Pandas
    
    Parameters
    ---------
    df:      Pandas dataframe
    col:    Target column if any  
    method:  String {hist | histkde | kde}, default phik
    """
    dftmp = df.copy()
    sp = dftmp.select_dtypes("number").shape[1]*0.5
    if col:
        corr = dftmp.corr(method=method)[col].sort_values(ascending=True).to_frame()
        plt.figure(figsize=(2,sp))
    else:
        corr = dftmp.corr(method=method)
        plt.figure(figsize=(sp*2,sp))
    sns.heatmap(corr,annot=True,mask=None if col else np.triu(corr))
    plt.show()

def df_corrphik(df,col=False):
    """Correlation plot using phik 
    
    Parameters
    ---------
    df:    Pandas dataframe
    col:  Target column if any  
    """
    dftmp = df.copy()
    corr = dftmp.phik_matrix().round(2)
    #gcorr = df.global_phik()
    #sigf = df.significance_matrix().round(decimals = 1)
    sp = dftmp.shape[1]*0.5
    
    if col:
        corr = corr[col].sort_values(ascending=True).to_frame()
        plt.figure(figsize=(2,sp))
    else:
        plt.figure(figsize=(sp*2,sp))
    sns.heatmap(corr,annot=True,mask=None if col else np.triu(corr))
    plt.show()
    
def df_calc_vif(df,features=False):
    """Calculation of VIF (Variance Inflation Factor) factors for selected features. 
    
    Parameters
    ---------
    df:        Pandas dataframe
    features:  Feature columns to be included. Note that non-numerical columns will be ignored
    """
    vif = {}   
    num = df.select_dtypes(exclude=["bool","object","category"]).columns.to_list()
    cols = [c for c in features if c in num] if features else num
    
    for c in cols:
        X = df[[f for f in cols if f != c]]
        y = df[c]
        r2 = LinearRegression().fit(X,y).score(X,y) # Scoring by R2
        vif[c] = 1/(1-r2) if r2 != 1 else 99999999
        
    print("Guide to interpreting VIF values:")
    print("VIF = 1: Features are not correlated")
    print("1 < VIF < 5: Features are moderately correlated")
    print("VIF > 5: Features are highly correlated")
    print("VIF > 10: High correlation between features and is cause for concern")
    return pd.DataFrame({"VIF":vif})
    
def df_numdevplot(df,col,step=20):
    """Stackplot of selected numerical columns (col) 
    
    Parameters
    ---------
    df:   Pandas dataframe
    col:  Column name
    step: number of index steps to use when plotting growth  
    """
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    lb = [c for c in col if df[c].dtype != "O"]
    df2 = df[lb].divide(df[lb].sum(axis = 1), axis=0 )
    x = df.index #.to_listt()
    y1 = [df[c] for c in lb]
    y2 = [df2[c] for c in lb] 
    y3 = [df[c].pct_change(periods=step) for c in lb]

    ax[0].stackplot(x, y1,labels = lb)
    ax[1].stackplot(x, y2,labels = lb)
    for i,y in enumerate(y3):
        ax[2].plot(x,y,label=lb[i])
        
    ax[0].set_ylabel("Sum",fontsize=15)
    ax[1].set_ylabel("Percent of sales",fontsize=15)
    ax[2].set_ylabel(f"Growth [%] per {step} steps",fontsize=15)

    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower left")
    plt.show()      
    
def df_catnum(df,col):
    """Creates a copy of the inserted dataframe with values in the selected columns turned into integers.
    
    Parameters
    ---------
    df:   Pandas dataframe
    col:  Column name
    """
    ut = df.copy()
    dctu = {v:i for i,v in enumerate(ut[col].unique())} 
    ut[col] = ut[col].map(dctu).astype("int")
    return ut,dctu

def df_allcatnum(df):
    ut = df.copy()
    
    for col in ut.select_dtypes(exclude=["number"]).columns:
        ut[col] = ut[col].map({v:i for i,v in enumerate(ut[col].unique())}).astype("int")
    return ut

def df_downsample(df,col):
    """Downsampling to balanse dataset 'df' on column 'col'.
    
    Parameters
    ---------
    df:   Pandas dataframe
    col:  Name of column to be used to balance the dataframe
    """
    dfut = pd.DataFrame(columns=df.columns)
    ant = df[col].isna().sum()
    
    if ant>0:
        print(f"{ant} NaN-values in target column, downsampling not performed.")
    else:
        nr = df[col].value_counts().min()
        for n in df[col].unique():
            dfut=dfut.append(df.loc[df[col]==n].sample(n=nr))
    return dfut