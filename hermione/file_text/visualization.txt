import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from yellowbrick.target import FeatureCorrelation

class Visualization:
    
    @staticmethod
    def general_analysis(df):
        """
        Plot function general analysis of graphs

        Parameters
        ----------    
        df : pd.DataFrame
             dataframe to be analyzed
 
        Returns
        ----
        None
        """
        pass
    
    @staticmethod
    def missing_analysis(df):
        """
        Function plots the percentage of missings in all columns of the DataFrame
    
        Parameters
        ----------    
        df : pd.DataFrame
             dataframe on which the missing will be analyzed
             
        Returns
        -------
        None
        """
        df_isnull = (df.isnull().sum() / len(df))*100
        df_isnull = df_isnull.drop(df_isnull[df_isnull ==0].index).sort_values(ascending = False)
        missing_data = pd.DataFrame({'Percentual Missing': df_isnull})
        missing_data.plot.bar()
    
    @staticmethod
    def count_values(df, feature, title):
        """
        Plot of count of distinct values ​​of a feature
    
        Parameters
        ----------    
        df      : pd.DataFrame
                  dataframe with the values
        feature : str 
                  name of the feature to be counted
        title   : str 
                  chart title
        
        Returns
        ----
        None
        """
        g = sns.catplot(feature, data=df, aspect=4, kind="count")
        g.set_xticklabels(rotation=90)
        g = plt.title(title)
        
    @staticmethod
    def regression_analysis( y_true, y_pred, path=None):
        """
        Analysis of the real and predicted y of the regression model
    
        Parameters
        ----------    
        y_true      : array
                      true values
        y_pred      : array
                      predicted values
        path        : str
                      path where the graphics will be saved
        
        Returns
        -------
        None
        """
        residual = y_true - y_pred
        print("Histogram")
        Visualization.histogram(residual, "Residual")
        print("Scatter")
        Visualization.scatter(y_pred, residual, "pred", "residual", path=path)
        print("Scatter")
        Visualization.scatter(y_true, y_pred, "y_test", "pred", path=path)

    @staticmethod
    def histogram(values, title, fig_size=(4,3), path=None):
        """
        Histogram plot of a set of values
    
        Parameters
        ----------    
        values      : array
                      values
        title       : str
                      title
        fig_size    : tuple
                      figure size
        path        : str
                      path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        plt.clf()
        f, ax = plt.subplots(1, figsize=fig_size)
        ax.hist(values, bins=60)
        ax.set_title(title)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/hist_'+title+'.png')
    
        
    @staticmethod
    def correlation_analysis(df, fig_size=(5,4), path=None):
        """
        Correlation of variables in the dataframe
    
        Parameters
        ----------    
        df       : pd.DataFrame
                   dataframe
        fig_size : tuple
                   figure size
        path     : str
                   path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        plt.clf()
        f, ax = plt.subplots(1, figsize=fig_size)
        corr = round(df.corr(), 4)
        sns.heatmap(corr, 
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values, ax=ax)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/correlation.png')

    @staticmethod    
    def features_correlation(df, cols, target, fig_size=(6,6), path=None):
        """
        Correlation of variables in the dataframe with respect to the target
    
        Parameters
        ----------    
        df       : pd.Dataframe
                   dataframe with the data to calculate the correlation
        cols     : array
                   columns to be correlated with the target
        target   : str
                   target name
        fig_size : tuple
                   figure size
        path     : str
                   path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        f, ax = plt.subplots(1, figsize=fig_size)
        ax.set_xlabel("Feature Correlation")
        visualizer = FeatureCorrelation(labels=list(cols))
        visualizer.fit(df[cols], df[target])
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/features_correlation.png')

    @staticmethod
    def scatter(x, y, xlabel, ylabel, fig_size=(5,4), groups=None, group_color=None, path=None):
        """
        Plot scatter
    
        Parameters
        ----------    
        x            : array
                       list of x-axis values
        y            : array
                       list of y-axis values
        x_label      : str
                       label x
        y_label      : array
                       label y                                    
        fig_size     : tuple
                       figure size
        groups       : array
                       group list
        group_color  : dict
                       group colors
        path         : str
                       path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        f, ax = plt.subplots(1, figsize=fig_size)
        sns.scatterplot(x, y, hue=groups, palette=group_color, legend="full", ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/scatter_'+xlabel+'_'+ylabel+'.png')
            
    @staticmethod
    def bar(x, y, xlabel, ylabel, fig_size=(5,4), est=np.mean, groups=None, group_color=None, path=None):
        """
        Plot bar
    
        Parameters
        ----------    
        x           : array
                      list of x-axis values
        y           : array
                      list of y-axis values                            
        x_label     : str
                      label x
        y_label     : array
                      label y                                    
        fig_size    : tuple
                      figure size
        est         : np
                      numpy function for aggregating the bars
        groups      : array
                      group list
        group_color : dict
                      group colors
        path        : str
                      path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        f, ax = plt.subplots(1, figsize=fig_size)
        sns.barplot(x, y, ax=ax, hue=groups, estimator=est, color=group_color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/barr_'+xlabel+'_'+ylabel+'.png')

    @staticmethod
    def line(x, y, xlabel, ylabel, fig_size=(5,4), est=np.mean, groups=None, group_color=None, path=None):
        """
        Plot bar
    
        Parameters
        ----------    
        x         : array
                    list of x-axis values
        y         : array
                    list of y-axis values                            
        x_label   : str
                    label x
        y_label   : array
                    label y                                    
        fig_size  : tuple
                    figure size
        est       : np
                    numpy function for aggregating the bars
        groups    : array
                    group list
        group_color : dict
                    group colors
        path      : str
                    path where the graphics will be saved
                      
        Returns
        -------
        None
        """
        f, ax = plt.subplots(1, figsize=fig_size)
        sns.lineplot(x, y, hue=groups, estimator=est, color=group_color, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/linha_'+xlabel+'_'+ylabel+'.png')
            
    @staticmethod
    def box_plot(x, y, xlabel, ylabel, fig_size=(5,4), path=None):
        """
        Plot line
    
        Parameters
        ----------    
        x         : array
                    list of x-axis values
        y         : array
                    list of y-axis values                            
        x_label   : str
                    label x
        y_label   : array
                    label y                                  
        fig_size  : tuple
                    figure size
        path      : str
                    path where the graphics will be saved
    
        Returns
        -------
        None
        """
        f, ax = plt.subplots(1, figsize=fig_size)
        sns.boxplot(x=x, y=y, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        f.tight_layout()
        if(path != None):
            f.savefig(path+'/boxplot_'+xlabel+'_'+ylabel+'.png')

    @staticmethod
    def scatter_interactive(df, col_name_x,  col_name_y, xlabel, ylabel, hover, fig_size=(400,300), **kwargs):
        """
        Interactive plotter
    
        Parameters
        ----------
        df          : pd.Dataframe
                      dataframe
        col_name_x  : str
                      col name in x
        col_name_y  : str
                      col name in y                        
        x_label     : str
                      label x
        y_label     : str
                      label y      
        hover       : list
                      values show when pass mouse
        fig_size    : tuple
                      figure size
        **kwargs    : **kwargs
                      to inform other properties of the chart. For example, 
                      to set the color to a type, just pass color = "blue"
        Returns
        -------
        None
        """
        alt.Chart(df, width=fig_size[0], height=fig_size[1]).mark_circle().encode(
                    alt.X(col_name_x, title=xlabel),
                    alt.Y(col_name_y, title=ylabel),
                    tooltip=hover,
                    **kwargs
                ).interactive().display()

    @staticmethod
    def bar_interactive(df, col_name_x, col_name_y, xlabel, ylabel, hover, fig_size=(400,300), **kwargs):
        """
        Interactive plotter
    
        Parameters
        ----------
        df          : pd.Dataframe
                      dataframe
        col_name_x  : str
                      col name in x
        col_name_y  : str
                      col name in y                        
        x_label     : str
                      label x
        y_label     : str
                      label y      
        hover       : list
                      values show when pass mouse
        fig_size    : tuple
                      figure size
        **kwargs    : **kwargs
                      to inform other properties of the chart. For example, 
                      to set the color to a type, just pass color = "blue"
        Returns
        -------
        None
        """
        alt.Chart(df, width=fig_size[0], height=fig_size[1]).mark_bar().encode(
                    alt.X(col_name_x, title=xlabel),
                    alt.Y(col_name_y, title=ylabel),
                    tooltip=hover,
                    **kwargs
                ).interactive().display()
        
    @staticmethod
    def line_interactive(df, col_name_x, col_name_y, xlabel, ylabel, hover, fig_size=(400,300), **kwargs):
        """
        Interactive plotter
    
        Parameters
        ----------
        df          : pd.Dataframe
                      dataframe
        col_name_x  : str
                      col name in x
        col_name_y  : str
                      col name in y                        
        x_label     : str
                      label x
        y_label     : str
                      label y      
        hover       : list
                      values show when pass mouse
        fig_size    : tuple
                      figure size
        **kwargs    : **kwargs
                      to inform other properties of the chart. For example, 
                      to set the color to a type, just pass color = "blue"
        Returns
        -------
        None
        """
        alt.Chart(df, width=fig_size[0], height=fig_size[1]).mark_line().encode(
                    alt.X(col_name_x, title=xlabel),
                    alt.Y(col_name_y, title=ylabel),
                    tooltip=hover,
                    **kwargs
                ).interactive().display()
    
