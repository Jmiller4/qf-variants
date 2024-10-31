

import pandas as pd

def simulate_1p1v(donation_df):
	# each person casts their one vote for the project they donated the most to.
	# funds are awarded proportional to the amount of votes a project gets.

	projects = donation_df.columns

	first_choices = donation_df.idxmax(axis=1)

	vote_tallies = {p: len(first_choices[first_choices == p]) for p in projects}

	total_votes = sum(vote_tallies[p] for p in projects)

	results = {p : vote_tallies[p] / total_votes for p in projects}

	return results


def simulate_1t1v(donation_df, strategy='shade'):
	
	# When simulating 1t1v, we need to consider the possibility that people donated the way they did because they were donating to a QF mechanism.
	# If they were directly donating, they might've donated differently.

	# The "strategy" variable denotes different ways of considering that counterfactual.

	# if strategy='none', just assume people would've donated exactly the same. 
	# if strategy='shade', assume people are shading down their donations for QF. So, if they donated x under QF, they would've donated x^2 under 1t1v
	# if strategy='boost', assume people are donating more since it's QF. So, if they donated x under QF, they would've donated sqrt(x) under 1t1v.

	# IMO shade is the most likely counterfactual but the other ones are in there just in case folks want to explore.

	assert strategy in ['none', 'shade', 'boost']

	if strategy == 'shade':
		donation_df = donation_df ** 2

	if strategy == 'boost':
		donation_df = donation_df ** 0.5

	projects = donation_df.columns
	
	vote_tallies = {p: donation_df[p].sum() for p in projects}

	total_votes = sum(vote_tallies[p] for p in projects)

	results = {p : vote_tallies[p] / total_votes for p in projects}

	return results


def simulate_borda_count(donation_df):
	projects = donation_df.columns
	num_proj = len(projects)

	point_tallies = {p: 0 for p in projects}

	for i in donation_df.index:
		row = donation_df.loc[i].copy()
		for j in range(num_proj):
			current_max = row.idxmax()
			point_tallies[current_max] += num_proj - (j+1)
			row.drop([current_max])

	total_points = sum(point_tallies[p] for p in projects)

	results = {p : point_tallies[p] / total_points for p in projects}

	return results

def simulate_borda_count(donation_df):
	projects = donation_df.columns
	num_proj = len(projects)

	point_tallies = {p: 0 for p in projects}

	for i in donation_df.index:
		ranking = donation_df.loc[i].rank()
		for p in projects:
			point_tallies[p] += ranking[p] - 1

	total_points = sum(point_tallies[p] for p in projects)

	results = {p : point_tallies[p] / total_points for p in projects}

	return results
