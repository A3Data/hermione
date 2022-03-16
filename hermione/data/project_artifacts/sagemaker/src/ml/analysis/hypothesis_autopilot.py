import pandas as pd
import numpy as np
import operator
import warnings
from ml.analysis.hypothesis_testing import HypothesisTester


class HTestAutoPilot:
    """
    Chooses what is the most adequate hypothesis test for a given dataset,
    based on its datatypes and the assumptions of each test
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
            if data not in [0, 1]:
                return False
        return True

    @staticmethod
    def check_norm(sample1, sample2, alpha=0.05, normality_method="shapiro"):
        """
        Check normality

        Parameters
        ----------
        sample1 : array_like
                Array of sample data, must be quantitative data.
        sample2 : array_like
                Array of sample data, must be quantitative data.
        alpha : float
                level of significance (default = 0.05)
        normality_method : string
                           normality test to be applied

        Returns
        -------
        Array
        """
        return [
            HypothesisTester.normality_test(
                s, alpha=alpha, method=normality_method, show_graph=False
            ).loc["normal"][0]
            for s in [sample1, sample2]
        ]

    @staticmethod
    def correlation(
        sample1,
        sample2,
        alpha=0.05,
        alternative="two-sided",
        normality_method="shapiro",
        show_graph=True,
        **kwargs
    ):
        """
        Autopilot for correlation tests

        Parameters
        ----------
        sample1 : array_like
                Array of sample data, must be quantitative data.
        sample2 : array_like
                Array of sample data, must be quantitative data.
        alpha : float
                level of significance (default = 0.05)
        alternative : string
                Specify whether the alternative hypothesis is `'two-sided'`,
                `'greater'` or `'less'` to specify the direction of the
                test.
        normality_method : string
                           normality test to be applied

        Returns
        -------
        pd.DataFrame
        """
        sample1, sample2 = np.array(sample1), np.array(sample2)
        np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32, np.float64]]
        if any([t not in np_types for t in [sample1.dtype, sample2.dtype]]):
            raise Exception(
                "Samples are not numerical. ",
                "Try using categorical_test method instead.",
            )

        check_bin1 = HTestAutoPilot.check_binary(sample1)
        check_bin2 = HTestAutoPilot.check_binary(sample2)

        if check_bin1 and check_bin2:
            raise Exception(
                "Both samples are binary, ", "unable to calculate correlation."
            )
        elif sum([check_bin1, check_bin2]) == 1:
            print(
                "One binary sample and one real sample.",
                "Point-biserial correlation is going to be applied.",
            )
            corr_method = "pointbiserial"
            binary_sample = sample2 if not check_bin1 else sample1
            num_sample = sample1 if not check_bin1 else sample2
            sample1, sample2 = [binary_sample, num_sample]
        else:
            check_norm1, check_norm2 = HTestAutoPilot.check_norm(
                sample1, sample2, alpha, normality_method
            )
            if check_norm1 and check_norm2:
                print("Samples are normally distributed.", "Using Pearson correlation.")
                corr_method = "pearson"
            else:
                print(
                    "Samples are not normally distributed.",
                    "Using Spearman correlation.",
                )
                corr_method = "spearman"
        df_result = HypothesisTester.correlation_test(
            sample1,
            sample2,
            method=corr_method,
            alpha=alpha,
            alternative=alternative,
            show_graph=show_graph,
            **kwargs
        )
        return df_result

    @staticmethod
    def categorical(
        df,
        sample1,
        sample2,
        alpha=0.05,
        alternative="two-sided",
        correction=True,
        show_graph=True,
        **kwargs
    ):
        """
        Autopilot for tests with categorical variables

        Parameters
        ----------
        df : pandas.DataFrame
                The dataframe containing the ocurrences for the test.
        sample1 : string
                The variable name for the test. Must be names of columns in
                ``data``.
        sample2 : string
                The variable name for the test. Must be names of columns in
                ``data``.
        alpha : float
                level of significance (default = 0.05)
        alternative : string
                        Specify whether to return `'two-sided'`, `'greater'` or
                        `'less'` p-value to specify the direction of the test.
        correction : bool
                Whether to apply Yates' correction when the degree of freedom
                of the observed contingency table is 1 (Yates 1934). In case
                of Chi-squared test.
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        df_chi2 = HypothesisTester.chi2_test(
            df, sample1, sample2, correction, alpha, show_graph, **kwargs
        )
        table = df.groupby([sample1, sample2]).size() > 5
        if table.sum() == len(table):
            df_result = df_chi2
        else:
            if len(df[sample1].unique()) == 2 and len(df[sample2].unique()) == 2:
                warnings.warn(
                    "The number of observations is not indicated "
                    + "for the chi-squared test, cannot garantee a "
                    + "correct inference. Also using Fisher's exact"
                    + " test."
                )
                df_fisher = HypothesisTester.fisher_exact_test(
                    df, sample1, sample2, alpha, show_graph=False
                )
                df_result = pd.concat([df_chi2, df_fisher], axis=1).fillna("-")
            else:
                warnings.warn(
                    "The number of observations is not indicated "
                    + "for the chi-squared test, cannot garantee a "
                    + "correct inference."
                )
                df_result = df_chi2
        return df_result

    @staticmethod
    def independent_difference(
        sample1,
        sample2,
        alpha=0.05,
        alternative="two-sided",
        correction="auto",
        r=0.707,
        normality_method="shapiro",
        show_graph=True,
        **kwargs
    ):
        """
        Autopilot for testing the difference in means for independent samples

        Parameters
        ----------
        sample1 : array_like
                Array of sample data, must be quantitative data.
        sample2 : array_like
                Array of sample data, must be quantitative data.
        alpha : float
                level of significance (default = 0.05)
        alternative : string
                    Specify whether the alternative hypothesis is
                    `'two-sided'`, `'greater'` or `'less'` to specify
                    the direction of the test.
        correction : string or boolean
                    For unpaired two sample T-tests, specify whether
                    or not to correct for unequal variances using Welch
                    separate variances T-test. If 'auto', it will automatically
                    uses Welch T-test when the sample sizes are unequal, as
                    recommended by Zimmerman 2004.
        r : float
            Cauchy scale factor for computing the Bayes Factor.
            Smaller values of r (e.g. 0.5), may be appropriate when small
            effect sizes are expected a priori; larger values of r are
            appropriate when large effect sizes are expected
            (Rouder et al 2009). The default is 0.707 (= :math:`\sqrt{2} / 2`).
        normality_method : string
                normality test to be applied
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        check_norm1, check_norm2 = HTestAutoPilot.check_norm(
            sample1, sample2, alpha, normality_method
        )
        if check_norm1 and check_norm2:
            print(
                "Samples are normally distributed, an ideal condition",
                "for the application of t-test",
            )
            df_result = HypothesisTester.t_test(
                sample1,
                sample2,
                paired=False,
                alpha=alpha,
                alternative=alternative,
                correction=correction,
                r=r,
                show_graph=show_graph,
                **kwargs
            )
        elif (check_norm1 is False and len(sample1) < 30) or (
            check_norm2 is False and len(sample2) < 30
        ):
            print(
                "At least one of the samples is not normally distributed.",
                "However, the t-test can be applied due to central limit",
                "theorem (n>30). The Mann-Whitney test is also an option",
                "as it does not make any assumptions about data ditribution",
                "(non-parametric alternative)",
            )
            df_result = HypothesisTester.mann_whitney_2indep(
                sample1, sample2, alpha, alternative, show_graph, **kwargs
            )
        else:
            print(
                "At least one of the samples is not normally distributed",
                "and due to the number of observations the central limit",
                "theorem does not apply. In this case, the Mann-Whitney",
                "test is used as it does not make any assumptions about",
                "data ditribution (non-parametric alternative)",
            )
            df_result = HypothesisTester.t_test(
                sample1,
                sample2,
                paired=False,
                alpha=alpha,
                alternative=alternative,
                correction=correction,
                r=r,
                show_graph=show_graph,
                **kwargs
            )
            df_result_non_param = HypothesisTester.mann_whitney_2indep(
                sample1, sample2, alpha, alternative, show_graph=False
            )
            df_result = (
                pd.concat([df_result, df_result_non_param], axis=1)
                .reindex(
                    [
                        "T",
                        "dof",
                        "cohen-d",
                        "BF10",
                        "power",
                        "U-val",
                        "RBC",
                        "CLES",
                        "p-val",
                        "CI95%",
                        "H0",
                        "H1",
                        "Result",
                    ]
                )
                .fillna("-")
            )
        return df_result

    @staticmethod
    def dependent_difference(
        sample1,
        sample2,
        alpha=0.05,
        alternative="two-sided",
        correction="auto",
        r=0.707,
        normality_method="shapiro",
        show_graph=True,
        **kwargs
    ):
        """
        Autopilot for testing the difference in means for dependent samples

        Parameters
        ----------
        sample1 : array_like
                Array of sample data, must be quantitative data.
        sample2 : array_like
                Array of sample data, must be quantitative data.
        alpha : float
                level of significance (default = 0.05)
        alternative : string
                    Specify whether the alternative hypothesis is
                    `'two-sided'`, `'greater'` or `'less'` to
                    specify the direction of the test.
        correction : string or boolean
                    For unpaired two sample T-tests, specify whether
                    or not to correct for unequal variances using
                    Welch separate variances T-test. If 'auto', it
                    will automatically uses Welch T-test when the
                    sample sizes are unequal, as recommended by Zimmerman
                    2004.
        r : float
            Cauchy scale factor for computing the Bayes Factor.
            Smaller values of r (e.g. 0.5), may be appropriate
            when small effect sizes are expected a priori; larger
            values of r are appropriate when large effect sizes are
            expected (Rouder et al 2009).
            The default is 0.707 (= :math:`\sqrt{2} / 2`).
        normality_method : string
                normality test to be applied
        show_graph: boolean
                    display the graph.

        Returns
        -------
        pd.DataFrame
        """
        diff_sample = sorted(list(map(operator.sub, sample1, sample2)))
        check_norm_diff = HypothesisTester.normality_test(
            diff_sample, alpha, normality_method, show_graph=False
        ).loc["normal"][0]

        if check_norm_diff:
            print(
                "The distribution of differences is normally distributed",
                "an ideal condition for the application of t-test.",
            )
            df_result = HypothesisTester.t_test(
                sample1,
                sample2,
                paired=True,
                alpha=alpha,
                alternative=alternative,
                correction=correction,
                r=r,
                show_graph=show_graph,
                **kwargs
            )
        elif len(sample1) > 30 and len(sample2) > 30:
            print(
                "The distribution of differences is not normally",
                "distributed. However, the t-test can be applied",
                "due to central limit theorem (n>30). The Wilcoxon",
                "test is also an option as it does not make any assumptions",
                "about data ditribution (non-parametric alternative).",
            )
            df_result = HypothesisTester.t_test(
                sample1,
                sample2,
                paired=True,
                alpha=alpha,
                alternative=alternative,
                correction=correction,
                r=r,
                show_graph=show_graph,
                **kwargs
            )
            df_result_non_param = HypothesisTester.wilcoxon_test(
                sample1, sample2, alpha, alternative, show_graph=False, **kwargs
            )
            df_result = (
                pd.concat([df_result, df_result_non_param], axis=1)
                .reindex(
                    [
                        "T",
                        "dof",
                        "cohen-d",
                        "BF10",
                        "power",
                        "W-val",
                        "RBC",
                        "CLES",
                        "p-val",
                        "CI95%",
                        "H0",
                        "H1",
                        "Result",
                    ]
                )
                .fillna("-")
            )
        else:
            print(
                "The distribution of differences is not normally",
                "distributed and due to the number of observations the",
                "central limit theorem does not apply. In this case,",
                "the Wilcoxon test is indicated as it does not make",
                "any assumptions about data distribution",
                "(non-parametric alternative).",
            )
            df_result = HypothesisTester.wilcoxon_test(
                sample1, sample2, alpha, alternative, show_graph, **kwargs
            )
        return df_result
