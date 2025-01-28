import scipy.stats
import numpy as np

################ HYPOTHESIS TESTS: ################
###### FOR COMPARING TWO MODELS (OBJECT DETECTION) ###########
def WilcoxonSignedRankTwoSided(x,y = None,alpha = 0.05):
    """
        The function used is wilcoxon from scipy.stats, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        for documentation. The purpose of x and y is the same.
        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value


        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that x and y are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class. (Class decisions.)



        The function performs the Wilcoxon Signed Rank Test on paired data to test the null hypothesis that the median difference between pairs is zero. 
        This is a two-sided test, meaning it tests for deviations in both directions (greater or smaller differences from zero).
        Assumptions: 

        Independence: The differences between the paired observations should be independent. 
        Each pair should represent an independent observation.

        Ordinal or Continuous Data: The data should be at least ordinal, meaning the values have an inherent order. 
        Continuous data are typically used in this test, but ordinal data may also be valid in certain cases.

        Symmetry of the Distribution of Differences: Although the Wilcoxon Signed Rank Test does not require the differences to follow a normal distribution, it assumes that the distribution of the differences is symmetric about the median. 
        If the distribution is heavily skewed, the test may be less reliable.

        No Extreme Outliers: The test assumes that the data do not contain extreme outliers that could heavily influence the test results, 
        as this can distort the signed ranks.
    """
    zstatistic, pvalue = scipy.stats.wilcoxon(x,y)
    if pvalue <= alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, models are different")
    else:
        print("null hypothesis is not rejected")
    return zstatistic, pvalue

############### FOR COMPARING SEVERAL MODELS (OBJECT DETECTION) ############
def FriedmansTest(*samples,alpha = 0.05):
    """
        The function used is friedmanchisquare from scipy.stats, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html
        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value


        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that the samples are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class. (Class decisions.)



        The Friedman Test evaluates whether there are differences in treatments or conditions across multiple related groups (e.g., repeated measurements on the same subjects).
        It is an alternative to the one-way repeated measures ANOVA when the assumptions of normality are violated.
        The null hypothesis of the Friedman Test is that the distributions of the groups are identical, 
        while the alternative hypothesis suggests that at least one group has a different distribution.

        Null Hypothesis: There is no difference in the distributions of the groups (inferences).
        Alternative Hypothesis: At least one of the groups (inferences) is different from the others.

        Assumptions: 

        Paired or Repeated Measurements: The data must consist of repeated measurements on the same subjects or items under different conditions (treatments, time points, etc.). 
        Each subject or item must experience all conditions or treatments.

        Ordinal or Continuous Data: The data must be at least ordinal, meaning the values have a meaningful order. 
        While continuous data are most common, ordinal data can also be used as long as there is a clear ranking.

        Nominal Number of Groups: The test assumes that the number of groups is fixed and predefined. 
        The Friedman Test requires at least three conditions (or time points) to be meaningful.

        At least three data samples must be given.
    """
    zstatistic, pvalue = scipy.stats.friedmanchisquare(*samples)
    if pvalue <= alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, at least one model differs significantly")
    else:
        print("null hypothesis is not rejected")
    return zstatistic,pvalue

################## CHECKING VARIANCES ########################
######### WITH NORMALITY ASSUMPTION TESTS: ############
##### COMPARING TWO MODELS ONLY ######
def Ftest(x,y,alpha=0.05):
    """
        The F-test compares the variances of two datasets by calculating the ratio of their variances. 
        If the two datasets come from populations with the same variance, the ratio of the variances should be close to 1.
        The F-test is most commonly used in the context of comparing multiple models (ANOVA) or to check the homogeneity of variance assumption in regression models.

        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value

        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that x and y are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class. (Class decisions.)

        Null Hypothesis: The two population variances are equal.
        Alternative Hypothesis: The two population variances are not equal (for a two-tailed test) or one is greater than the other (for a one-tailed test).

        Assumptions:

        Independence: The data samples must be independent.
        Each observation in one sample should not be influenced by observations in the other sample.

        Normality: Both groups (inferences) being compared should be approximately normally distributed. The F-test assumes normality of the underlying distributions of the two datasets. 
        The test is sensitive to deviations from normality, especially with small sample sizes.

        Homogeneity of Variance: In the context of ANOVA, the F-test assumes that the variances of different groups are equal..
    """
    # Computing Variances of x and y
    var1 = np.var(x,ddof=1)
    var2 = np.var(y,ddof=1)

    # Computing the F-statistic
    f_statistic = var1/var2

    # Degrees of freedom:
    dof1=len(x)-1
    dof2=len(y)-1
    
    # Calculating the p-value
    p_value = 1-scipy.stats.f.cdf(f_statistic,dof1,dof2)

    if p_value<=alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, the variances are significantly different")
    else:
        print("null hypothesis is not rejected")
    
    return f_statistic,p_value

####### COMPARING MORE THAN TWO MODELS ########
def bartlett(*samples,alpha=0.05):
    """
        The function used is bartlett from scipy.stats, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html
        The Bartlett test checks if k samples (inferences) come from populations with the same variance. 
        It is commonly used as a preliminary test for homogeneity of variances when performing one-way ANOVA or other statistical analyses that assume equal variances among groups.

        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value
            
        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that the samples are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class. (Class decisions.)

        Null Hypothesis: The population variances of all groups are equal.
        Alternative Hypothesis: At least one of the groups has a different population variance.

        Assumptions:

        Independence: The observations within each sample should be independent of each other.

        Normality: The Bartlett test assumes that the data within each group are normally distributed. 
        If the data significantly deviate from normality, the Bartlett test may give inaccurate results, especially for small sample sizes. For non-normal data, alternative tests like the Levene test might be more appropriate.
    """
    statistic,p_value = scipy.stats.bartlett(*samples)
    if p_value<=alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, all models have equal variances")
    else:
        print("null hypothesis is not rejected")
    return statistic,p_value

##################### WITH NO NORMALITY ASSUMPTION ################
######## FOR TWO OR MORE MODELS #############
def LevenesTest(*samples, alpha = 0.05):
    """ 
        The function used is levene from scipy.stats, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
        Levene’s Test evaluates whether multiple models have equal variances by testing the null hypothesis that the population variances are equal across the inferences. Unlike Bartlett's test, Levene's test is less sensitive to deviations from normality, making it more suitable for non-normal data distributions.
        Levene’s test is widely used in situations where you want to check the assumption of equal variances before performing ANOVA or other statistical tests that assume homogeneity of variances.

        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value

        Say that you have classes 0,1,2,3,...,9.
        Then it is assumed that the samples are in the foramt of a list in which each cell's value (class number) corresponds to the data sample in the dataset.
        For example [0,8,3,1,3,7,3,9,6,4,5,3,2,...,2]. each cell is a class. (Class decisions.)

        Null Hypothesis: The population variances of all groups are equal (the variances are homogeneous).
        Alternative Hypothesis: At least one group has a different population variance (the variances are not equal).

        Assumptions: 

        Independence: The observations within each group must be independent of one another.

        Scale of Measurement: The data should be at least ordinal (although continuous data are more common in practice).

        Notes:

        Levene's test is less sensitive to non-normality than Bartlett’s test.
        Therefore, it can handle non-normal distributions better than other tests of equal variance.

        The test can handle unequal sample groups' sizes, 
        unlike some other variance tests that may be sensitive to this condition.
    """
    statistic, pvalue = scipy.stats.levene(*samples)
    # idea for checking variance and hypothesis tests overall: conduct tests on multiple test datasets
    if pvalue<= alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, at least one model has a variance different from the others")
    else:
        print("null hypothesis is not rejected")
    return statistic, pvalue


############# CHECKING FOR NORMALITY ###########
def ShapiroWilkTest(x, alpha=0.05):
    """
        The function used is shapiro from scipy.stats, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
        The Shapiro-Wilk test evaluates the null hypothesis that the data is normally distributed. If the p-value from the test is less than a chosen significance level (usually 0.05),
        we reject the null hypothesis and conclude that the data significantly deviates from a normal distribution. Otherwise, if the p-value is greater than the significance level, 
        we fail to reject the null hypothesis and conclude that there is no significant evidence that the data is non-normal.

        The Significance Level (alpha) chosen is 0.05
        The function returns the tuple z-statistic, p-value

        Null Hypothesis: The data follows a normal distribution.
        Alternative Hypothesis: The data does not follow a normal distribution.

        Assumptions:

        Independence: The observations in the dataset must be independent of each other.

        Normality of Data: The Shapiro-Wilk test specifically tests for normality. It assumes that the dataset comes from a normal distribution. 
        If the p-value is low, it indicates that the data does not follow a normal distribution.

        Note:

        The Shapiro-Wilk test is most effective with small to moderate sample sizes (less than 50 observations, 
        though it can be used with larger samples as well).
    """
    statistic, pvalue = scipy.stats.shapiro(x)
    if pvalue<=alpha:
        print("pvalue <= alpha hence null hypothesis is rejected, data is not normally distributed")
    else:
        print("null hypothesis is not rejected")
    return statistic,pvalue

#BrownForsythe?
#AndersonDarling?
#KolmogorovSmirnov?
#Log-Loss?
