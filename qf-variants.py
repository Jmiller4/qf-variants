import pandas as pd
from itertools import combinations
from math import log
from math import sqrt
from math import floor
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import time
import json
import os
def add(x,y):
  return x + y

#
#
# definitions for a *bunch* of variations of QF
#
# in all of these functions,
# - donation_df is expected to be a pandas dataframe where rows are wallets, columns are projects, and entries represent a wallet's total donation amount to a project
# - cluster_df is expected to be a pandas dataframe where rows are wallets, columns are cluster, and entries are denote the strength of a user's membership in that cluster.
#
# also important to note: these functions all return the matching amounts each project should get under that variant of QF -- to get the full funding amount,
# you need to add in the direct donations as well!
#


# first, some helper functions
def binarize(df):
  return df.applymap(lambda x: 1 if x > 0 else 0)

def align(donation_df, cluster_df):
  # first, drop users who haven't made any donations / aren't in any clusters
  cluster_df.drop(cluster_df.index[cluster_df.apply(lambda row: all(row == 0), axis=1)],inplace=True)
  donation_df.drop(donation_df.index[donation_df.apply(lambda row: all(row == 0), axis=1)],inplace=True)

  # Also remove wallets that are just in one dataframe, but not the other
  cluster_df.drop(set(cluster_df.index) - set(donation_df.index), inplace=True)
  donation_df.drop(set(donation_df.index) - set(cluster_df.index), inplace=True)

  #make sure the indices are sorted the same way (important for making sure the matrix multiplications work later)
  cluster_df.sort_index(inplace=True)
  donation_df.sort_index(inplace=True)

  return donation_df, cluster_df


# now on to the QF variants

def standard_qf(donation_df):
  projects = donation_df.columns
  funding = {p: (donation_df[p].apply(lambda x: sqrt(x)).sum() ** 2) - donation_df[p].sum() for p in projects}

  return funding


def pairwise(donation_df, M=0.01):

  projects = donation_df.columns
  donors = donation_df.index

  # start off with funding = sum of individual donations, then add the pairwise matching amounts
  #funding = {p: donation_df[p].sum() for p in projects}
  funding = {p : 0 for p in projects}
  sqrt_donation_df = donation_df.apply(lambda col: np.sqrt(col))

  # The next line of code creates a matrix containing each pairwise coefficient k_i,j
  # In-depth expanation:
  # The dot product is a matrix multiplication that will give us a matrix where entry i,j is the dot product of
  # i's square-rooted donation vector with j's square-rooted donation vector.
  # Next, even though M is technically a scalar, pandas will automatically interpret the syntax "M + <matrix>"
  # by assuming that M here refers to a matrix with M in every entry, and the same dimensions as the actual matrix
  # on the other side of the +.
  # Same goes for "M / <matrix>".
  # The result is a matrix, "k_matrix", where entry i,j is the k_i,j described in the original pairwise matching blog post
  k_matrix = M / (M + sqrt_donation_df.dot(sqrt_donation_df.transpose()))

  proj_sets = {d : set([p for p in projects if donation_df.loc[d, p] > 0]) for d in donors}

  for  wallet1, wallet2 in combinations(donors,2):
    for p in proj_sets[wallet1].intersection(proj_sets[wallet2]):
      funding[p] += sqrt_donation_df.loc[wallet1, p] * sqrt_donation_df.loc[wallet2, p] * k_matrix.loc[wallet1, wallet2]

  return funding


def cluster_profile_pairwise(donation_df, cluster_df):

  cluster_df = binarize(cluster_df)

  donation_df, cluster_df = align(donation_df, cluster_df)

  projects = donation_df.columns
  donors = donation_df.index
  clusters = cluster_df.columns
  cluster_members = cluster_df.index

  # start off with funding = sum of individual donations, then add the pairwise matching amounts
  #funding = {p: donation_df[p].sum() for p in projects}
  funding = {p : 0 for p in projects}


  # the pairwise matching coefficient for agents i and j is:
  # (# groups just i is in + # groups just j is in) / (# groups i is in + # groups j is in)

  # first, make a matrix whose entries are the numerators of the above formula for every pair of agents
  # we make it by first setting each entry to be the total number of clusters, then subracting the clusters that both i and j are in,
  # then subtracting the clusters that neither i nor j are in. We're left with the clusters that exactly one of i or j are in.
  numerator_matrix = pd.DataFrame(index=donors, columns=donors, data=len(clusters)) - cluster_df.dot(cluster_df.transpose()) - ((1-cluster_df).dot(1-cluster_df.transpose()))


  # now we make a matrix C representing the denominators of the above formula
  # A is a vector where entry i is the number of groups i is in
  A = cluster_df.apply(sum, axis=1)
  # B is a matrix where every entry in row i is the number of groups i is in
  B = pd.DataFrame(index=donors,columns=donors,data=[A]*len(donors))
  # by adding B and its transpose, we get a matrix where entry (i,j) is the number of groups i is in + the number of groups j is in
  denominator_matrix = B + B.transpose()
  # finally, we can get the coefficient matrix by dividing the numerators by the denominators
  coeffs = numerator_matrix / denominator_matrix


  for p in projects:

    non_donors = donation_df[donation_df[p] == 0].index

    donor_only_donation_df = donation_df.drop(non_donors, axis=0)

    donor_only_coeffs = coeffs.drop(non_donors, axis=1).drop(non_donors, axis=0)

    y = donor_only_donation_df[p].apply(sqrt)
    z = pd.DataFrame(y)
    QF_matrix = z.dot(z.transpose())
    funding[p] += (QF_matrix * donor_only_coeffs).sum().sum()

  return funding

def clustermatch(donation_df, cluster_df):

  projects = donation_df.columns
  clusters = cluster_df.columns
  donors = donation_df.index
  cluster_members = cluster_df.index

  normalized_clusters = cluster_df.apply(lambda row: row / row.sum() if any(row) else 0, axis=1)

  donation_df.drop(list(set(donors) - set(cluster_members)), inplace=True)
  normalized_clusters.drop(list(set(cluster_members) - set(donors)), inplace=True)

  normalized_clusters.sort_index(inplace=True)
  donation_df.sort_index(inplace=True)

  B = donation_df.transpose().dot(normalized_clusters)

  # B should be a matrix where rows are projects, columns are clusters, and entry (i,j) is cluster j's donation to project i

  funding = {p: B.loc[p].apply(lambda x: sqrt(x)).sum() ** 2 - B.loc[p].sum() for p in projects}
  return funding


def donation_profile_clustermatch(donation_df):
  # run cluster match, using donation profiles as the clusters
  # i.e., everyone who donated to the same set of projects gets put under the same square root.

  # donation_df is expected to be a pandas Dataframe where rows are unique donors, columns are projects,
  # and entry i,j denote user i's total donation to project j

  # we'll store donation profiles as binary strings.
  # i.e. say there are four projects total. if an agent donated to project 0, project 1, and project 3, they will be put in cluster "1101".
  # here the indices 0,1,2,3 refer to the ordering in the input list of projects.

  projects = donation_df.columns
  don_profiles = donation_df.apply(lambda row: ''.join('1' if row[p] > 0 else '0' for p in projects), axis=1)

  don_profile_df = pd.DataFrame(index=donation_df.index, columns=don_profiles.unique(), data=0)

  for wallet in donation_df.index:
    don_profile_df.loc[wallet, don_profiles[wallet]] = 1

  return clustermatch(donation_df, don_profile_df)


def COCM(donation_df, cluster_df, fancy=True):
  # run CO-CM on a set of funding amounts and clusters
  # if "fancy" is false, follow the formula in the whitepaper exactly. If "fancy" is true, get fancy with it.

  # # first, drop users who haven't made any donations / aren't in any clusters
  # cluster_df.drop(cluster_df.index[cluster_df.apply(lambda row: all(row == 0), axis=1)],inplace=True)
  # donation_df.drop(donation_df.index[donation_df.apply(lambda row: all(row == 0), axis=1)],inplace=True)

  # # Also remove wallets that are just in one dataframe, but not the other
  # cluster_df.drop(set(cluster_df.index) - set(donation_df.index), inplace=True)
  # donation_df.drop(set(donation_df.index) - set(cluster_df.index), inplace=True)

  # #make sure the indices are sorted the same way (important for making sure the matrix multiplications work later)
  # cluster_df.sort_index(inplace=True)
  # donation_df.sort_index(inplace=True)

  donation_df, cluster_df = align(donation_df, cluster_df)

  projects = donation_df.columns
  clusters = cluster_df.columns
  donors = donation_df.index
  cluster_members = cluster_df.index


  # normalize the cluster dataframe so that rows sum to 1. Now, an entry tells us the "weight" that a particular cluster has for a particular user.
  # if a user is in 0 clusters, their row will be a bunch of NaNs if we naively divide by 1.
  # we shouldn't have any such users anyways, but just in case, we'll fill such a row with 0s instead
  normalized_clusters = cluster_df.apply(lambda row: row / row.sum() if any(row) else 0, axis=1)

  binarized_clusters = binarize(cluster_df)

  if fancy:
    # friendship_matrix is a matrix whose rows and columns are both wallets,
    # and a value of 1 at index i,j means that wallets i and j are in at least one cluster together.
    friendship_matrix = cluster_df.dot(cluster_df.transpose()).apply(lambda col: col > 0)

    # k_indicators is a dataframe with wallets as rows and clusters as columns.
    # if wallet i is not in cluster g, then entry i,g is is the fraction of i's friends who are in cluster g (i's friends are the agents i is in a shared cluster with).
    # if wallet i is in cluster g, then entry i,g is 1.

    # in the past, we used cluster_df in the following line instead of binarized_clusters
    k_indicators = friendship_matrix.dot(binarized_clusters).apply(lambda row: row / friendship_matrix.loc[row.name].sum(), axis=1)
    # ... and the following line used cluster_df instead of binarized_clusters
    k_indicators = k_indicators.apply(lambda row: np.maximum(row, binarized_clusters.loc[row.name]), axis=1)

  else:

    # friendship_matrix is a matrix whose rows and columns are both wallets,
    # and a value greater than 0 at index i,j means that wallets i and j are in at least one group together.
    friendship_matrix = cluster_df.dot(cluster_df.transpose())

    # k_indicators is a dataframe with wallets as rows and stamps as columns.
    # entry i,g is True if wallet i is in a shared group with anyone from g, and False otherwise.
    k_indicators = friendship_matrix.dot(cluster_df).apply(lambda col: col > 0)

  # Create a dictionary to store funding amounts for each project.
  # first we'll fund each project with the sum of donations to that project
  # then we'll add in the pairwise matching amounts, which is the hard part.
  #funding = {p: donation_df[p].sum() for p in projects}
  funding = {p: 0 for p in projects}

  for p in projects:
    # get the actual k values for this project using contributions and indicators.

    # C will be used to build the matrix of k values.
    # It is a matrix where rows are wallets, columns are clusters, and the ith row of the matrix just has wallet i's contribution to the project in every entry.
    C = pd.DataFrame(index=donors, columns = ['_'], data = donation_df[p].values).dot(pd.DataFrame(index= ['_'], columns = clusters, data=1))
    # C is attained by taking the matrix multiplication of the column vector donation_df[p] (which is every agent's donation to project p) and a row vector with as many columns as projects, and a 1 in every entry
    # the above line is so long mainly because you need to cast Pandas series' (i.e. vectors) as dataframes (i.e. matrices) for the matrix multiplication to work.

    # now, K is a matrix where rows are wallets, columns are projects, and entry i,g ranges between c_i and sqrt(c_i) depending on i's relationship with cluster g and whether "fancy" was set to true or not.
    K = (k_indicators * C.pow(1/2)) + ((1 - k_indicators) * C)


    # Now we have all the k values, which are one of the items inside the innermost sum expressed in COCM.
    # the other component of these sums is a division of each k value by the number of groups that user is in.
    # P_prime is a matrix that combines k values and total group memberships to attain the value inside the aforementioned innermost sum.
    # In other words, entry g,h of P_prime is:
    #
    #       sum_{i in g} K(i,h) / T_i
    #
    # where T_i is the total number of groups that i is in
    P_prime = K.transpose().dot(normalized_clusters)

    # Now, we can create P_prime, whose non-diagonal entries g,h represent the pairwise subsidy given to the pair of groups g and h.
    P = (P_prime * P_prime.transpose()).pow(1/2)

    # The diagonal entries of P are not relevant, so get rid of them. We only care about the pairwise subsidies between distinct groups.
    np.fill_diagonal(P.values, 0)

    # Now the sum of every entry in P is the amount of subsidy funding COCM awards to the project.
    funding[p] += P.sum().sum()


  return funding


def standard_donation(donation_df):
  # just do a normal vote (nothing quadratic)
  projects = donation_df.columns
  funding = {p: donation_df[p].sum() for p in projects}
  return funding

