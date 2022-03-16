import pandas as pd
import numpy as np
import pingouin as pg
import operator
from scipy.stats import fisher_exact, pointbiserialr
from scipy.stats import norm
from ..visualization import Visualizer


class HypothesisTester:
    """
    Class to perform hypothesis test
    """

    @staticmethod
    def define_hypothesis(df, statistic, alternative, paired, alpha):
        """
        Defines the null and alternative hypothesis.

        Parameters
        ----------
        df : pd.DataFrame
              test result dataframe.
        statistic : string
                    'mean' or 'median' depending on the test.
        alternative : string
                      Specify whether the alternative hypothesis is
                      `'two-sided'`, `'greater'` or `'less'` to specify the
                      direction of the test. `'greater'` tests the alternative
                      that ``x`` has a larger mean than ``y``.
        paired: boolean
                Specify whether the two observations are related
                (i.e. repeated measures) or independent.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)

        Returns
        -------
        pd.DataFrame
        """
        paired_text = (
            f"the {statistic} difference" if paired else f"difference in {statistic}"
        )
        hypothesis = {
            "two-sided_H0": f"{paired_text} equal to zero",
            "two-sided_H1": f"{paired_text} not equal to zero",
            "greater_H0": f"{paired_text} greater than or equal to zero",
            "greater_H1": f"{paired_text} less than zero",
            "less_H0": f"{paired_text} less than or equal to zero",
            "less_H1": f"{paired_text} greater than zero",
        }
        df = HypothesisTester.test_alternative(df, hypothesis, alternative, alpha)
        return df

    @staticmethod
    def test_alternative(df, hypothesis, alternative="two-sided", alpha=0.05):
        """
        Tests the hypothesis using the p-value and adds the conclusion
        to the results DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
              test result dataframe.
        hypothesis: dict
                Specify the H0 and H1 hypothesis.
        alternative : string
                      Specify whether the alternative hypothesis is
                      `'two-sided'`, `'greater'` or `'less'` to specify
                      the direction of the test.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)

        Returns
        -------
        pd.DataFrame
        """
        df["H0"] = hypothesis[alternative + "_H0"]
        df["H1"] = hypothesis[alternative + "_H1"]
        formatted_alpha = round(alpha * 100, 2)
        conclusion = (
            "There is no evidence" if df["p-val"][0] > alpha else "There is evidence"
        )
        df[
            "Result"
        ] = f"{conclusion} to reject the null hypothesis at {formatted_alpha}% significance"
        return df

    @staticmethod
    def correlation_test(
        sample1,
        sample2,
        method="pearson",
        alpha=0.05,
        alternative="two-sided",
        show_graph=True,
        **kwargs,
    ):
        """
        Perform correlation between two variables.

        Parameters
        ----------
        sample1 : array_like
                  First set of observations. ``sample1`` must be independent.
        sample2 : array_like
                  Second set of observations. ``sample2`` must be independent.
        method: string
              Correlation type: pearson, spearman, kendall, bicor, percbend,
              shepherd, skipped or pointbiserial.
        alpha : float
              level of significance for the confidence intervals
              (default = 0.05)
        alternative : string
                    Specify whether the alternative hypothesis is
                    `'two-sided'`, `'greater'` or `'less'` to specify
                    the direction of the test.
        show_graph: boolean
                  display the graph.

        Returns
        -------
        pd.DataFrame
        """
        text = "relationship between the two variables"
        hypothesis = {
            "two-sided_H0": f"there is no {text}",
            "two-sided_H1": f"there is a {text}",
            "greater_H0": f"there is no positive {text}",
            "greater_H1": f"there is a positive {text}",
            "less_H0": f"there is no negative {text}",
            "less_H1": f"there is a negative {text}",
        }
        if method == "pointbiserial":
            pb_corr = pointbiserialr(sample1, sample2)
            df = pd.DataFrame(
                data={"r": [pb_corr.correlation], "p-val": [pb_corr.pvalue]}
            )
            df = df.rename({0: "pointbiserial"})
        else:
            df = pg.corr(x=sample1, y=sample2, alternative=alternative, method=method)
        if show_graph:
            Visualizer.scatter(x=sample1, y=sample2, **kwargs)
        return HypothesisTester.test_alternative(df, hypothesis, alternative, alpha).T

    @staticmethod
    def normality_test(sample, alpha=0.05, method="shapiro", show_graph=True, **kwargs):
        """
        Tests the null hypothesis that the data is normally distributed

        Parameters
        ----------
        sample : array_like
                 Array of sample data.
        alpha : float
                level of significance (default = 0.05)
        method : string
                 normality test to be applied.
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        hypothesis = {
            "two-sided_H0": f"the data is normally distributed",
            "two-sided_H1": f"the data is not normally distributed",
        }
        sample = np.array(sample)
        np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32, np.float64]]
        sample_dtypes = sample.dtype
        if any([t not in np_types for t in [sample_dtypes]]):
            raise Exception(
                "Samples are not numerical. Try using",
                "categorical_test method instead.",
            )
        df = pg.normality(sample, method=method)
        df.rename(columns={"pval": "p-val"}, index={0: "Normality"}, inplace=True)
        if show_graph:
            Visualizer.qqplot(sample, **kwargs)
        return HypothesisTester.test_alternative(df, hypothesis, alpha=alpha).T

    @staticmethod
    def fisher_exact_test(df, sample1, sample2, alpha=0.05, show_graph=True, **kwargs):
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

        Returns
        -------
        pd.DataFrame
        """
        hypothesis = {
            "two-sided_H0": "the samples are independent",
            "two-sided_H1": "the samples are dependent",
        }
        table = pd.crosstab(df[sample1], df[sample2])
        statistic, p_value = fisher_exact(table, "two-sided")
        df_result = pd.DataFrame(
            data={"statistic": [statistic], "p-val": [p_value]}
        ).rename({0: "fisher exact"})
        if show_graph:
            pd.crosstab(df[sample1], df[sample2], normalize="index").plot(
                kind="bar", color=["r", "b"], **kwargs
            )
        return HypothesisTester.test_alternative(
            df=df_result, hypothesis=hypothesis, alpha=alpha
        ).T

    @staticmethod
    def chi2_test(
        df, sample1, sample2, correction=True, alpha=0.05, show_graph=True, **kwargs
    ):
        """
        Chi-squared independence test between two categorical variables.

        Parameters
        ----------
        df : pandas.DataFrame
              The dataframe containing the ocurrences for the test.
        sample1 : string
                The variable name for the test. Must be name of column
                in ``data``.
        sample2 : string
                The variable name for the test. Must be name of column
                in ``data``.
        correction : bool
                Whether to apply Yates' correction when the degree of freedom
                of the observed contingency table is 1 (Yates 1934).
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05).
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        hypothesis = {
            "two-sided_H0": "the samples are independent",
            "two-sided_H1": "the samples are dependent",
        }
        expected, observed, stats = pg.chi2_independence(
            df, sample1, sample2, correction
        )
        p_value = stats.loc[stats["test"] == "pearson"]["pval"][0]
        statistic = stats.loc[stats["test"] == "pearson"]["chi2"][0]
        df_result = pd.DataFrame(
            data={"statistic": [statistic], "p-val": [p_value]}
        ).rename({0: "chi2"})
        df_result["Expected Distribution"] = str(expected.values.tolist())
        df_result["Observed Distribution"] = str(observed.values.tolist())
        if show_graph:
            pd.crosstab(df[sample1], df[sample2], normalize="index").plot(
                kind="bar", color=["r", "b"], **kwargs
            )
        return HypothesisTester.test_alternative(df_result, hypothesis, alpha=alpha).T

    @staticmethod
    def t_test(
        sample1,
        sample2,
        paired=False,
        alpha=0.05,
        alternative="two-sided",
        correction="auto",
        r=0.707,
        show_graph=True,
        **kwargs,
    ):
        """
        T-test can be paired or not. The paired t-test compares the means
        of the same group or item under two separate scenarios.
        The unpaired t-test compares the means of two independent groups.

        Parameters
        ----------
        sample1 : array_like
                  First set of observations.
        sample2 : array_like or float
                  Second set of observations. If ``sample2`` is a single
                  value, a one-sample T-test is computed against that value.
        paired: boolean
                Specify whether the two observations are related
                (i.e. repeated measures) or independent.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)
        alternative : string
                      Specify whether the alternative hypothesis is
                      `'two-sided'`, `'greater'` or `'less'` to specify the
                      direction of the test.
        correction : string or boolean
                      For unpaired two sample T-tests, specify whether or not
                      to correct for unequal variances using Welch separate
                      variances T-test. If 'auto', it will automatically uses
                      Welch T-test when the sample sizes are unequal, as
                      recommended by Zimmerman 2004.
        r : float
            Cauchy scale factor for computing the Bayes Factor. Smaller values
            of r (e.g. 0.5), may be appropriate when small effect sizes are
            expected a priori; larger values of r are appropriate when large
            effect sizes are expected (Rouder et al 2009).
            The default is 0.707 (= :math:`\sqrt{2} / 2`).
        show_graph: boolean
                    display the graph.

        Returns
        -------
          pd.DataFrame
        """
        confidence = 1 - alpha
        df_result = pg.ttest(
            sample1,
            sample2,
            paired=paired,
            confidence=confidence,
            alternative=alternative,
            correction=correction,
            r=r,
        )
        if show_graph:
            if paired:
                difference = [x - y for x, y in zip(sample1, sample2)]
                Visualizer.histogram(difference, **kwargs)
            else:
                Visualizer.density_plot(sample1, sample2, fig_size=(5, 4), **kwargs)
        return HypothesisTester.define_hypothesis(
            df_result, "mean", alternative, paired, alpha
        ).T

    @staticmethod
    def non_param_unpaired_ci(sample1, sample2, alpha=0.05):
        """
        Confidence interval for differences (non-Gaussian unpaired data)

        Parameters
        ----------
        sample1 : array_like
                First set of observations. ``sample1`` must be independent.
        sample2 : array_like
                Second set of observations. ``sample2`` must be independent.
        alpha : float
              level of significance for the confidence intervals
              (default = 0.05)

        Returns
        -------
        float
        """
        n1 = len(sample1)
        n2 = len(sample2)
        N = norm.ppf(1 - alpha / 2)
        diffs = sorted([i - j for i in sample1 for j in sample2])
        k = np.math.ceil(n1 * n2 / 2 - (N * (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5))
        CI = (round(diffs[k - 1], 3), round(diffs[len(diffs) - k], 3))
        return CI

    @staticmethod
    def mann_whitney_2indep(
        sample1, sample2, alpha=0.05, alternative="two-sided", show_graph=True, **kwargs
    ):
        """
        Mann-Whitney U Test (Wilcoxon rank-sum test): A nonparametric test to
        compare the medians between two independent groups.
        It is the non-parametric version of the independent T-test.

        Parameters
        -----------
        sample1 : array_like
                First set of observations. ``sample1`` must be independent.
        sample2 : array_like
                Second set of observations. ``sample2`` must be independent.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)
        alternative : string
                      Specify whether to return `'two-sided'`, `'greater'` or
                      `'less'` p-value to specify the direction of the test.
        show_graph: boolean
                    display the graph

        Returns
        -------
        pd.DataFrame
        """
        df_result = pg.mwu(sample1, sample2, alternative=alternative)
        if alternative == "two-sided":
            ci = HypothesisTester.non_param_unpaired_ci(sample1, sample2, alpha)
            df_result["CI" + str(int((1 - alpha) * 100)) + "%"] = str([ci[0], ci[1]])
        if show_graph:
            Visualizer.density_plot(sample1, sample2, **kwargs)
        return HypothesisTester.define_hypothesis(
            df_result, "median", alternative, paired=False, alpha=alpha
        ).T

    @staticmethod
    def non_param_paired_ci(sample1, sample2, alpha):
        """
        Confidence interval for differences between the two samples.

        Parameters
        -----------
        sample1 : array_like
                First set of observations. ``sample1`` must be independent.
        sample2 : array_like
                Second set of observations. ``sample2`` must be independent.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)

        Returns
        -------
        float
        """
        n = len(sample1)
        N = norm.ppf(1 - alpha / 2)
        diff_sample = sorted(list(map(operator.sub, sample2, sample1)))
        averages = sorted(
            [
                (s1 + s2) / 2
                for i, s1 in enumerate(diff_sample)
                for _, s2 in enumerate(diff_sample[i:])
            ]
        )
        k = np.math.ceil(
            n * (n + 1) / 4 - (N * (n * (n + 1) * (2 * n + 1) / 24) ** 0.5)
        )
        CI = (round(averages[k - 1], 3), round(averages[len(averages) - k], 3))
        return CI

    @staticmethod
    def wilcoxon_test(
        sample1, sample2, alpha=0.05, alternative="two-sided", show_graph=True, **kwargs
    ):
        """
        Wilcoxon Test: A nonparametric test to compare the medians between two
        dependent groups. It is the non-parametric version of the paired T-test

        Parameters
        ----------
        sample1 : array_like
                        First set of observations. ``sample1`` must be
                        related (e.g repeated measures) and, therefore, have
                        the same number of samples. Note that a listwise
                        deletion of missing values is automatically applied.
        sample2 : array_like
                        Second set of observations. ``sample2`` must be
                        related (e.g repeated measures) and, therefore, have
                        the same number of samples. Note that a listwise
                        deletion of missing values is automatically applied.
        alpha : float
                level of significance for the confidence intervals
                (default = 0.05)
        alternative : string
                      Specify whether to return `'two-sided'`, `'greater'` or
                      `'less'` p-value to specify the direction of the test.
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        df_result = pg.wilcoxon(sample1, sample2, alternative=alternative)
        if alternative == "two-sided":
            ci = HypothesisTester.non_param_paired_ci(sample1, sample2, alpha)
            df_result["CI" + str(int((1 - alpha) * 100)) + "%"] = str([ci[0], ci[1]])
        if show_graph:
            diff_sample = sorted(list(map(operator.sub, sample1, sample2)))
            Visualizer.histogram(diff_sample, **kwargs)
        return HypothesisTester.define_hypothesis(
            df_result, "median", alternative, paired=True, alpha=alpha
        ).T
