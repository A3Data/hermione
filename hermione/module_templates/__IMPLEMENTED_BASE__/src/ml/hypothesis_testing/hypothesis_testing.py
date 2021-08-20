import pandas as pd
import numpy as np
import pingouin as pg
import operator
from scipy.stats import fisher_exact
from scipy.stats import norm
from ml.visualization.visualization import Visualization


class Hypothesis_testing:
  """
  Class to perform hypothesis test
  """

  @staticmethod
  def define_hypothesis(df, statistic, alternative, paired, alpha):
    """
    Define the null and alternative hypothesis.

    Parameters
    ----------            
    df : pd.DataFrame
          test result dataframe.
    statistic : string
                'mean' or 'median' depending on the test.
    paired: boolean
            Specify whether the two observations are related
            (i.e. repeated measures) or independent.
    alpha : float
            level of significance for the confidence intervals (default = 0.05)
    alternative : string
                  Specify whether the alternative hypothesis is `'two-sided'`,
                  `'greater'` or `'less'` to specify the direction of the test.
                  `'greater'` tests the alternative that ``x`` has a larger
                  mean than ``y``.
    Returns
    -------
    pd.DataFrame
    """
    hypothesis_paired= {
      'two-sided_H0': f"the {statistic} difference equal to zero",
      'two-sided_H1': f"the {statistic} difference not equal to zero",
      'greater_H0': f"the {statistic} difference greater than or equal to zero",
      'greater_H1': f"the {statistic} difference less than zero",
      'less_H0': f"the {statistic} difference less than or equal to zero",
      'less_H1': f"the {statistic} difference greater than zero"
    }

    hypothesis_not_paired= {
      'two-sided_H0': f"difference in {statistic} equal to zero",
      'two-sided_H1': f"difference in {statistic} not equal to zero",
      'greater_H0': f"difference in {statistic} greater than or equal to zero",
      'greater_H1': f"difference in {statistic} less than zero",
      'less_H0': f"difference in {statistic} less than or equal to zero",
      'less_H1': f"difference in {statistic} greater than zero"
    }
    if paired:
      df = Hypothesis_testing.test_alternative(df, hypothesis_paired, alternative, alpha)
    else:
      df = Hypothesis_testing.test_alternative(df, hypothesis_not_paired, alternative, alpha)
    return df

  @staticmethod
  def test_alternative(df, hypothesis, alternative='two-sided',
                       alpha=0.05):
    """
    Test alternative to define H0 and H1.
    Validate if the hypothesis was reject or not.

    Parameters
    ----------            
    df : pd.DataFrame
          test result dataframe.
    statistic : string
                'mean' or 'median' depending on the test.
    paired: boolean
            Specify whether the two observations are related
            (i.e. repeated measures) or independent.
    alpha : float
            level of significance for the confidence intervals (default = 0.05)
    alternative : string
                  Specify whether the alternative hypothesis is `'two-sided'`,
                  `'greater'` or `'less'` to specify the direction of the test.
    Returns
    -------
    pd.DataFrame
    """
    if alternative == 'two-sided':
      df['H0'] = hypothesis['two-sided_H0']
      df['H1'] = hypothesis['two-sided_H1']
    elif alternative == 'greater':
      df['H0'] =  hypothesis['greater_H0']
      df['H1'] = hypothesis['greater_H1']
    else:
      df['H0'] =  hypothesis['less_H0']
      df['H1'] = hypothesis['less_H1']

    if df['p-val'][0] > alpha:  
      df['Result'] = f'not reject null hypothesis that the {df["H0"][0]}'
    else:
      df['Result'] = f'reject the null hypothesis that the {df["H0"][0]}'
    return df

  @staticmethod
  def correlation_test(sample1, sample2, method='pearson', alpha=0.05,
                       alternative='two-sided', show_graph=True, title='', 
                       label1='', label2=''):
    """
    Perform correlation between two variables.

    Parameters
    ----------            
      sample1, sample2 : array_like
                  First and second set of observations. ``sample1`` and
                  ``sample2`` must be independent.
      alpha : float
              level of significance for the confidence intervals
              (default = 0.05)
      method: string
              Correlation type: pearson, spearman, kendall, bicor, percbend,
              shepherd or skipped
      alternative : string
                    Specify whether the alternative hypothesis is `'two-sided'`,
                    `'greater'` or `'less'` to specify the direction of the
                    test.                              
      show_graph: boolean
                  display the graph. 
      title : string    
              Title of the graph.  
      label1 : string    
                x-axis label.  
      label2 : string    
                y-axis label. 
    Returns
    -------
      pd.DataFrame
    """
    text = 'relationship between the two variables'
    hypothesis = {
      'two-sided_H0': f"there is no {text}",
      'two-sided_H1': f"there is a {text}",
      'greater_H0': f"there is no positive {text}",
      'greater_H1': f"there is a positive {text}",
      'less_H0': f"there is no negative {text}",
      'less_H1': f"there is a negative {text}"
    }
    df = pg.corr(x=sample1, y=sample2, alternative=alternative, method=method)
    if show_graph:
      Visualization.scatter(x=sample1, y=sample2, xlabel=label1, ylabel=label2, title=title)
    return Hypothesis_testing.test_alternative(df, hypothesis, alternative, alpha).T

  @staticmethod
  def normality_test(sample, alpha=0.05, method='shapiro',
                     show_graph=True, title='', label1='', label2=''):
    """
    Tests the null hypothesis that the data was drawn from a normal distribution

    Parameters
    ----------            
    sample : array_like
             Array of sample data.
    alpha : float
            level of significance (default = 0.05)
    method : string
             normality test to be applied
    show_graph: boolean
                display the graph 
    title : string    
            Title of the graph.  
    label1 : string    
              x-axis label.  
    label2 : string    
              y-axis label. 
                
    Returns
    -------
    pd.DataFrame
    """
    hypothesis = {
    'two-sided_H0': f"the data was drawn from a normal distribution",
    'two-sided_H1': f"the data was not drawn from a normal distribution"
    }
    sample = np.array(sample)
    np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32,
                                      np.float64]]
    sample_dtypes = sample.dtype
    if any([not t in np_types for t in [sample_dtypes]]):
        raise Exception('Non numerical variables... Try using categorical_test \
                         method instead.')
    df = pg.normality(sample)
    df.rename(columns={'pval': 'p-val'}, index={0: 'Normality'}, inplace=True)
    if show_graph:
      Visualization.qqplot(sample)
    return Hypothesis_testing.test_alternative(df, hypothesis, alpha=alpha).T

  @staticmethod
  def fisher_exact_test(df, sample1, sample2, alpha=0.05,
                        alternative='two-sided', show_graph=True, title='',
                        label1='', label2=''):
    """
    Perform a Fisher exact test.

    Parameters
    ----------
      df : pandas.DataFrame
            The dataframe containing the ocurrences for the test.
      sample1, sample2 : string
              The variables names for the test. Must be names of columns
              in ``data``.
      alpha : float
              level of significance for the confidence intervals
              (default = 0.05)
      alternative : string
                    Specify whether to return `'two-sided'`, `'greater'` or 
                    `'less'` p-value to specify the direction of the test.
      show_graph: boolean
                  display the graph. 
      title : string    
              Title of the graph.  
      label1 : string    
                x-axis label.  
      label2 : string    
                y-axis label. 
    Returns
    -------
      pd.DataFrame
    """
    ##MELHORAR
    text = 'between the samples'
    hypothesis = {
      'two-sided_H0': f"there is evidence of dependency {text}",
      'two-sided_H1': f"there is no evidence of dependency {text}",
      'greater_H0': f"there is no positive relationship {text}",
      'greater_H1': f"there is a positive relationship {text}",
      'less_H0': f"there is no negative relationship {text}",
      'less_H1': f"there is a negative relationship {text}"
    }
    table = pd.crosstab(df[sample1], df[sample2])
    statistic, p_value = fisher_exact(table, alternative)
    df_result = pd.DataFrame([( statistic, p_value)],
                             columns = ['statistic', 'p-val'])
    if show_graph:
      pd.crosstab(df[sample1], df[sample2],
                  normalize='index').plot(kind='bar',color=['r','b'])

    return Hypothesis_testing.test_alternative(df_result, hypothesis, alternative,
                                 alpha).T.rename(columns={0: 'fisher exact'})

  @staticmethod
  def chi2_test(df, sample1, sample2, correction=True, alpha=0.05,
                  show_graph=True, title='', label1='', label2=''):
    """
    Chi-squared independence tests between two categorical variables.

    Parameters
    ----------
    df : pandas.DataFrame
          The dataframe containing the ocurrences for the test.
    sample1, sample2 : string
            The variables names for the test. Must be names of columns
            in ``data``.
    correction : bool
            Whether to apply Yates' correction when the degree of freedom
            of the observed contingency table is 1 (Yates 1934).    
    alpha : float
            level of significance for the confidence intervals
            (default = 0.05).
    show_graph: boolean
                display the graph. 
    title : string    
            Title of the graph.  
    label1 : string    
              x-axis label.  
    label2 : string    
              y-axis label. 
    Returns
    -------
      pd.DataFrame
    """ 
    ##MELHORAR
    hypothesis = {
      'two-sided_H0': f"there is evidence of dependency between the samples",
      'two-sided_H1': f"there is no evidence of dependency between the samples"
    }
    expected, observed, stats = pg.chi2_independence(df, sample1, sample2,
                                                     correction)
    p_value = stats.loc[stats['test'] == 'pearson']['pval'][0]
    statistic = stats.loc[stats['test'] == 'pearson']['chi2'][0]
    df_result = pd.DataFrame([(statistic, p_value)], columns = ['statistic',
                                                                'p-val'])
    df_result['Expected Distribution'] = str(expected.values.tolist())
    df_result['Observed Distribution'] = str(observed.values.tolist())
    if show_graph:
      pd.crosstab(df[sample1], df[sample2],
                  normalize='index').plot(kind='bar', color=['r','b'])
    return Hypothesis_testing.test_alternative(df_result, hypothesis,
                                 alpha=alpha).T.rename(columns={0: 'chi2'})

  @staticmethod
  def t_test(sample1, sample2, paired=False, alpha=0.05,
             alternative='two-sided', correction='auto', r=0.707,
             show_graph=True, title='', label1='', label2=''):
    """
    T-test can be paired or not. The paired t-test compares the means of the
    same group or item under two separate scenarios. 
    The unpaired t-test compares the means of two independent groups. 

    Parameters
    ----------            
    sample1 : array_like
              First set of observations.
    sample2 : array_like or float
              Second set of observations. If ``sample2`` is a single value,
              a one-sample T-test is computed against that value.
    paired: boolean
            Specify whether the two observations are related
            (i.e. repeated measures) or independent.
    alpha : float
            level of significance for the confidence intervals (default = 0.05)
    alternative : string
                  Specify whether the alternative hypothesis is `'two-sided'`,
                  `'greater'` or `'less'` to specify the direction of the test.
    correction : string or boolean
                  For unpaired two sample T-tests, specify whether or not to
                  correct for unequal variances using Welch separate variances
                  T-test. If 'auto', it will automatically uses Welch T-test
                  when the sample sizes are unequal, as recommended by Zimmerman
                  2004.
    r : float
        Cauchy scale factor for computing the Bayes Factor.
        Smaller values of r (e.g. 0.5), may be appropriate when small effect
        sizes are expected a priori; larger values of r are appropriate when
        large effect sizes are expected (Rouder et al 2009).
        The default is 0.707 (= :math:`\sqrt{2} / 2`).
    show_graph: boolean
                display the graph 
    title : string    
            Title of the graph.  
    label1 : string    
              x-axis label.  
    label2 : string    
              y-axis label. 
    Returns
    -------
      pd.DataFrame
    """
    confidence = 1 - alpha   
    result = pg.ttest(sample1, sample2, paired=paired, confidence=confidence,
                      alternative=alternative, correction=correction, r=r)
    if show_graph: 
      if paired:
        difference = [x - y for x, y in zip(sample1, sample2)]
        Visualization.histogram(difference, title, label1, label2) 
      else:
        Visualization.density_plot(sample1, sample2, title, label1, label2, fig_size=(5,4))
    return Hypothesis_testing.define_hypothesis(result, 'mean', alternative, paired, alpha).T

  @staticmethod
  def non_param_unpaired_CI(sample1, sample2, alpha):
    """
    Confidence interval for differences (non-Gaussian unpaired data)
      
    Parameters
    ----------            
      sample1, sample2 : array_like
                First and second set of observations. ``sample1`` and 
                ``sample2`` must be independent.
      alpha : float
              level of significance for the confidence intervals
              (default = 0.05)
    Returns
    -------
    float
    """
    n1 = len(sample1)  
    n2 = len(sample2)        
    N = norm.ppf(1 - alpha/2) 
    diffs = sorted([i-j for i in sample1 for j in sample2])
    k = np.math.ceil(n1*n2/2 - (N * (n1*n2*(n1+n2+1)/12)**0.5))
    CI = (round(diffs[k-1],3), round(diffs[len(diffs)-k],3))
    return CI

  @staticmethod
  def mann_whitney_2indep(sample1, sample2, alpha=0.05,
                          alternative='two-sided', show_graph=True,
                          title='', label1='', label2=''):
    """
    Mann-Whitney U Test (Wilcoxon rank-sum test): A nonparametric test to
    compare the medians between two independent groups.
    It is the non-parametric version of the independent T-test.
      
    Parameters
    ----------            
    sample1, sample2 : array_like
              First and second set of observations. ``sample1`` and
              ``sample2`` must be independent.
    alpha : float
            level of significance for the confidence intervals
            (default = 0.05)
    alternative : string
                  Specify whether to return `'two-sided'`, `'greater'` or
                  `'less'` p-value to specify the direction of the test.
    show_graph: boolean
                display the graph
    title : string
            Title of the graph.
    label1 : string
              x-axis label.
    label2 : string
              y-axis label.
    Returns
    -------
    pd.DataFrame
    """
    result = pg.mwu(sample1, sample2, alternative=alternative)
    if alternative=='two-sided':
      ci = Hypothesis_testing.non_param_unpaired_CI(sample1, sample2, alpha)
      result['CI' + str(int((1-alpha)*100)) +'%'] = str([ci[0], ci[1]])
    if show_graph:
      Visualization.density_plot(sample1, sample2, title, label1, label2, fig_size=(5,4))
    return Hypothesis_testing.define_hypothesis(result, 'median', alternative, paired=False,
                                  alpha=alpha).T

  @staticmethod
  def non_param_paired_CI(list1, list2, alpha):
    """
    Confidence interval for differences between the two population.
      
    Parameters
    ----------            
    sample1, sample2 : array_like
              First and second set of observations. ``sample1`` and ``sample2``
              must be  independent.
    alpha : float
            level of significance for the confidence intervals (default = 0.05)
    Returns
    -------
    float
    """
    n = len(list1)      
    N = norm.ppf(1 - alpha/2) 
    diff_sample = sorted(list(map(operator.sub, list2, list1)))
    averages = sorted([(s1+s2)/2 for i, s1 in enumerate(diff_sample) 
                      for _, s2 in enumerate(diff_sample[i:])])
    k = np.math.ceil(n*(n+1)/4 - (N * (n*(n+1)*(2*n+1)/24)**0.5))
    CI = (round(averages[k-1],3), round(averages[len(averages)-k],3))
    return CI

  @staticmethod
  def wilcoxon_test(sample1, sample2, alpha=0.05, alternative='two-sided',
                    show_graph=True, title='', label1='', label2=''):
    """
    Wilcoxon: A nonparametric test to compare the medians between two
    dependent groups. It is the non-parametric version of the paired T-test

    Parameters
    ----------            
    sample1, sample2 : array_like
                        First and second set of observations. ``sample1`` and 
                        ``sample2`` must be related (e.g repeated measures) and,
                        therefore, have the same number of samples. Note that a
                        listwise deletion of missing values is automatically
                        applied.
    alpha : float
            level of significance for the confidence intervals (default = 0.05)
    alternative : string
                  Specify whether to return `'two-sided'`, `'greater'` or
                  `'less'` p-value to specify the direction of the test.
    show_graph: boolean
                display the graph.
    title : string    
            Title of the graph.  
    label1 : string    
              x-axis label.  
    label2 : string    
              y-axis label. 
    Returns
    -------
    pd.DataFrame
    """
    result = pg.wilcoxon(sample1, sample2, alternative=alternative)
    ci = Hypothesis_testing.non_param_paired_CI(sample1, sample2, alpha)
    result['CI' + str(int((1-alpha)*100)) +'%'] = str([ci[0], ci[1]])
    if show_graph:
      diff_sample = sorted(list(map(operator.sub, sample1, sample2)))
      Visualization.histogram(diff_sample, title, label1, label2, fig_size=(4,3))
    return Hypothesis_testing.define_hypothesis(result, 'median', alternative, paired=True,
                                  alpha=alpha).T