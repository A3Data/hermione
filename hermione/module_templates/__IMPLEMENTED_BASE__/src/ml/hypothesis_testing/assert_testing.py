import pandas as pd
import numpy as np
import operator
from ml.visualization.visualization import Visualization
from ml.hypothesis_testing.hypothesis_testing import Hypothesis_testing
import warnings

class Assert_hypothesis_test:
  """
  Apply best hypothesis following data propeties
  """

  @staticmethod
  def check_binary(col):
    """
    Check if data is binary

    Parameters
    ----------            
    col : array_like
              Array of sample data, must be quantitative data.
      
    Returns
    -------
    Bool
    """
    for data in col:
        if data != 0 and data !=1:
            return False
    return True

  @staticmethod
  def apply_correlation_test(sample1, sample2, alpha=0.05,
                             alternative='two-sided',
                             normality_method='shapiro', show_graph=True,
                             title='', label1='', label2=''):
    """
    Apply correlation test

    Parameters
    ----------            
    sample1 : array_like
              Array of sample data, must be quantitative data.
    sample2 : array_like
              Array of sample data, must be quantitative data.
    alpha : float
            level of significance (default = 0.05)
    test : string
              correlation test to be applied
    binary : string
            flag to identify if data is binary
    normality_method : string
              normality test to be applied
                
    Returns
    -------
    pd.DataFrame
    """
    sample1, sample2 = np.array(sample1), np.array(sample1) 
    np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32,
                                      np.float64]]
    sample1_dtypes, sample2_dtypes = sample1.dtype, sample2.dtype 
    if any([not t in np_types for t in [sample1_dtypes, sample2_dtypes]]):
        raise Exception('Non numerical variables. Try using categorical_test', \
                        'method instead.')

    check1 = Assert_hypothesis_test.check_binary(sample1)
    check2 = Assert_hypothesis_test.check_binary(sample2)

    if check1 and check2:
        print('Samples are binary, Pearson correlation is going to be', \
              'applied (Point-biserial).')
        ## fazer
    else:
      check1 = Hypothesis_testing.normality_test(sample1, alpha=alpha,
                                      method=normality_method,
                                      show_graph=False).loc['normal'][0]
      check2 = Hypothesis_testing.normality_test(sample2, alpha=alpha,
                                      method=normality_method,
                                      show_graph=False).loc['normal'][0]

      if check1 and check2:
          print('Samples have normal distribution.')
          df_result = Hypothesis_testing.correlation_test(sample1, sample2,
                                               method='pearson',
                                               alpha=alpha,
                                               alternative=alternative, 
                                               show_graph=show_graph,
                                               title=title,
                                               label1=label1, 
                                               label2=label2)
      else:
          print('Samples do not have normal distribution.')
          df_result = Hypothesis_testing.correlation_test(sample1, sample2,
                                               method='spearman',
                                               alpha=alpha,
                                               alternative=alternative,
                                               show_graph=show_graph,
                                               title=title,
                                               label1=label1,
                                               label2=label2)
    return df_result

  @staticmethod
  def apply_categorical_test(df, sample1, sample2, alpha=0.05,
                             alternative='two-sided', correction=True, 
                             show_graph=True, title='', label1='', label2=''):
    """
    Apply categorical test

    Parameters
    ----------
    df : pandas.DataFrame
            The dataframe containing the ocurrences for the test.
    sample1, sample2 : string
              The variables names for the test. Must be names of columns in
              ``data``.
    alpha : float
            level of significance (default = 0.05)
    alternative : string
                    Specify whether to return `'two-sided'`, `'greater'` or 
                    `'less'` p-value to specify the direction of the test.
    correction : bool
              Whether to apply Yates' correction when the degree of freedom of 
              the observed contingency table is 1 (Yates 1934). In case of
              Chi-squared test.   
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
    table = (df.groupby([sample1, sample2]).size() > 5)
    if table.sum() == len(table):
      df_result = Hypothesis_testing.chi2_test(df, sample1, sample2, correction, alpha,
              show_graph, title, label1, label2)
    else:
      df_chi2 = Hypothesis_testing.chi2_test(df, sample1, sample2, correction, alpha,
              show_graph, title, label1, label2)
      if len(df[sample1].unique()) == 2 and len(df[sample2].unique()) == 2:
        warnings.warn('The number of observations is not indicated for the', \
                      'chi-square test, an option is the Fisher\'s exact test.')
        df_fisher = Hypothesis_testing.fisher_exact_test(df, sample1, sample2, alpha,
                                              alternative, show_graph=False)
        df_result = pd.concat([df_chi2, df_fisher],axis=1)
      else:
        warnings.warn('The number of observations is not indicated for the', \
                      'chi-square test.')
        df_result = df_chi2
    return df_result

  @staticmethod
  def apply_independent_test(sample1, sample2, alpha=0.05,
                             alternative='two-sided', correction='auto',
                              r=0.707, normality_method='shapiro',
                             show_graph=True, title='', label1='', label2=''):
    """
    Assert independent test

    Parameters
    ----------
    df : pandas.DataFrame
            The dataframe containing the ocurrences for the test.
    sample1, sample2 : string
              The variables names for the test. Must be names of columns in
              ``data``.
    alpha : float
            level of significance (default = 0.05)
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
    normality_method : string
              normality test to be applied
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
    norm_sample_1 = Hypothesis_testing.normality_test(sample1, alpha,
                                                      normality_method,
                                                      show_graph=False
                                                      ).loc['normal'][0]
    norm_sample_2 = Hypothesis_testing.normality_test(sample2, alpha,
                                                      normality_method,
                                                      show_graph=False
                                                      ).loc['normal'][0]

    if norm_sample_1 and norm_sample_2:
      print('Samples have normal distribution, an ideal condition for the', \
            'application of t-test')
      result = Hypothesis_testing.t_test(sample1, sample2, paired=False, alpha=alpha,
                              alternative=alternative, correction=correction,
                              r=r, show_graph=show_graph, title=title,
                              label1=label1, label2=label2)
    elif (norm_sample_1==False and len(sample1)>30) or \
         (norm_sample_2==False and len(sample2)>30):
      print('At least one of the samples is not normally distributed.', \
            'However, the t-test can be applied due to central limit theorem', \
            '(n>30). The Mann-Whitney test is also an option as it does not', \
            'depend on the data distribution (non-parametric alternative)')
      result = Hypothesis_testing.t_test(sample1, sample2, paired=False, alpha=alpha,
                                         alternative=alternative, correction=correction,
                                         r=r, show_graph=show_graph, title=title,
                                         label1=label1, label2=label2)
      result_non_param = Hypothesis_testing.mann_whitney_2indep(sample1, sample2, alpha,
                                                                alternative,
                                                                show_graph=False,
                                                                title=title, label1=label1,
                                                                label2=label2)
      result = pd.concat([result,
                          result_non_param],
                         axis=1).reindex(['T', 'dof','cohen-d', 'BF10', 'power',
                                          'U-val', 'RBC', 'CLES', 'tail',
                                          'p-val', 'CI95%', 'H0', 'H1',
                                          'Result']).fillna('-')
    else:
      print('At least one of the samples is not normally distributed and due', \
            'to the number of observations the central limit theorem is not', \
            'indicated. In this case, the mann-whitney test is indicated as',
            'it does not depend on the data distribution (non-parametric', \
            'alternative)')
      result = Hypothesis_testing.mann_whitney_2indep(sample1, sample2, alpha, alternative,
                                           show_graph, title, label1, label2)
    return result

  @staticmethod
  def apply_dependent_test(sample1, sample2, alpha=0.05,
                           alternative='two-sided', correction='auto', r=0.707,
                           normality_method='shapiro', show_graph=True,
                           title='', label1='', label2=''):
    """
    Apply dependent test

    Parameters
    ----------
    df : pandas.DataFrame
            The dataframe containing the ocurrences for the test.
    sample1, sample2 : string
              The variables names for the test. Must be names of columns
              in ``data``.
    alpha : float
            level of significance (default = 0.05)
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
    normality_method : string
              normality test to be applied
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
    diff_sample = sorted(list(map(operator.sub, sample1, sample2)))
    normality_diff = Hypothesis_testing.normality_test(diff_sample, alpha,
                                            normality_method,
                                            show_graph=False).loc['normal'][0]

    if normality_diff:
      print('The distribution of differences is normally distributed', \
            'an ideal condition for the application of t-test.')
      result = Hypothesis_testing.t_test(sample1, sample2, paired=True, alpha=alpha,
                              alternative=alternative, correction=correction,
                              r=r, show_graph=show_graph, title=title,
                              label1=label1, label2=label2)
    elif len(sample1)>30 and len(sample2)>30:
      print('The distribution of differences is not normally distributed.', \
            'However, the t-test can be applied due to central limit theorem', \
            '(n>30). The Wilcoxon test is also an option as it does not', \
            'depend on the data distribution (non-parametric alternative).')
      result = Hypothesis_testing.t_test(sample1, sample2, paired=True, alpha=alpha,
                              alternative=alternative, correction=correction,
                              r=r, show_graph=show_graph, title=title,
                              label1=label1, label2=label2)
      result_non_param= Hypothesis_testing.wilcoxon_test(sample1, sample2, alpha,
                                              alternative, show_graph=False,
                                              title=title, label1=label1,
                                              label2=label2)
      result = pd.concat([result,
                          result_non_param],
                         axis=1).reindex(['T', 'dof', 'cohen-d', 'BF10',
                                          'power','W-val', 'RBC', 'CLES',
                                          'tail', 'p-val', 'CI95%', 'H0', 'H1',
                                          'Result']).fillna('-')
    else:
      print('The distribution of differences is not normally distributed and', \
            'due to the number of observations the central limit theorem is', \
            'not indicated. In this case, the Wilcoxon test is indicated as', \
            'it does not depend on the data distribution (non-parametric', \
            'alternative).')  
      result = Hypothesis_testing.wilcoxon_test(sample1, sample2, alpha, alternative,
                                     show_graph, title, label1, label2)
    return result