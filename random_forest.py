import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys

from Node import Node

def numeric2categoric(df,attr,q): #konwertacja ciaglych danych na kategorii
	df[attr] = pd.qcut(df[attr], q=q, labels=False)

def split_dataframe(df,split_at): # podzial danych na zbior trenujacy i testujacy
	df_train = df.sample(n=split_at)
	df_test = df.drop(df_train.index)
	return [df_train,df_test]

def get_predicted_value(prediction,index): #zwraca przewidywana wartosc klasy
	for i in prediction:
		if i[0] == index: #gdzie i[0] to index w zbiorze df
			return i[1] #i[1] jest przewidywana klasa

# ustawia wagi (wieksze wagi dla tych, na ktorych dotychczasowy model sie mylil)
# dodaje delta do elementow
# pozniej funckja sample w bagging normalizuje te wagi

def set_weights(df,weights,predictions,delta):
	wrong_predicts_indx = []

	if len(predictions) == 0:
		return weights
	else:
		for i in range(df.shape[0]):
			pr_val = get_predicted_value(predictions[-1],df.iloc[[i]].index[0])
			if df.iloc[[i]]['decision'].values[0] != pr_val:
				weights[i] += delta #dodanie wagi do elemenot
				wrong_predicts_indx.append(df.iloc[[i]].index[0])
	return 	weights


def bagging(df,weights): # metoda bootsrapowa tworzenia podziobiora ze zbioru trenujacego
	return df.sample(n=df.shape[0],replace=True,weights=weights)


# Entropy and Information Gain for ID3
def calculate_entropy(df):
	P = df[df['decision']==1].shape[0]
	N = df[df['decision']==2].shape[0]
	if P==0 or N==0:
		return 0
	return ((-P)/(P+N)*np.log2(P/(P+N)))-((N)/(P+N)*np.log2(N/(P+N)))

def select_node_infgain(df,attributes):
	infgain_sums = []
	for attr in attributes:
		attr_entropy = 0
		for ctg in df[attr].unique():
			inf_gain = calculate_entropy(df[df[attr]==ctg])
			attr_entropy += (df[df[attr]==ctg].shape[0]/df.shape[0])*inf_gain
		infgain_sums.append(calculate_entropy(df)-attr_entropy)
	return attributes[infgain_sums.index(max(infgain_sums))]


# GINI for CART
def calculate_gini(df):
	P = df[df['decision']==1].shape[0]
	N = df[df['decision']==2].shape[0]
	return 1-(np.square(P/(P+N))+np.square(N/(P+N)))

def select_node_gini(df,attributes):
	gini_sums = []
	for attr in attributes:
		sum = 0
		for ctg in df[attr].unique():
			sum += (df[df[attr]==ctg].shape[0]/df[attr].shape[0])*calculate_gini(df[df[attr]==ctg])
		gini_sums.append(sum)
	return attributes[gini_sums.index(min(gini_sums))]

def random_attributes(amount,attributes): #losowanie atrybutow dla kazdego wezla
	rand = random.sample(range(0,len(attributes)-1),amount)
	return [attributes[rand[i]] for i in range(amount)]

def check_leaves(df): # sprawdzenie czy mozna ustawic liscie
	P = df[df['decision']==1].shape[0]
	N = df[df['decision']==2].shape[0]
	if P == 0:
		return 2
	if N == 0:
		return 1
	return 0

def decision_tree(df_tree,amount,attributes):	#funckja tworzenia root i start tworzenia drzewa
	attrs = random_attributes(amount,attributes)
	root = select_node_gini(df_tree,attrs)

	categories = df_tree[root].unique()
	root_node = Node(root,categories)

	for i,ctg in enumerate(categories): # sprawdzenie wszystkich wartosci atrybutu
		split_tree(df_tree[df_tree[root]==ctg],
				   ctg,root_node,
				   amount,attributes)

	return root_node

def split_tree(df,ctg,parent_node,amount,attributes): # rekurencyjna funkcja tworzenia drzewa
	if check_leaves(df) == 2:
		parent_node.set_leave(ctg,2)
		return
	if check_leaves(df) == 1:
		parent_node.set_leave(ctg,1)
		return

	attrs = random_attributes(amount,attributes)
	node_attr = select_node_gini(df,attrs)

	child_node = Node(node_attr,df[node_attr].unique())
	parent_node.set_child(ctg,child_node)

	for ctg in child_node.values:
		split_tree(df[df[node_attr]==ctg],
				   ctg,child_node,
				   amount,attributes)

def test_tree(df,root_node): #testowanie drzewa
	predictions = []
	for i,row in df.iterrows():
		walk_tree(root_node,row,predictions,i)
	return predictions

def walk_tree(node,row,predictions,i): # rekurencyjne testowanie drzewa
	check_leave = node.get_class(row[node.attr])
	if check_leave == None:
		for val in node.values:
			if row[node.attr] == val:
				if node.next_node(row[node.attr]) != None:
					walk_tree(node.next_node(row[node.attr]),row,predictions,i)
	else:
		predictions.append([i,check_leave])
		return check_leave

def ensemble_voting(df,predictions): # glosowanie poszczegolnych drzew
	votes_table = []
	for i,row in df.iterrows():
		votes = []
		for pr in predictions:
			pr_val = get_predicted_value(pr,i)
			if pr_val != None:
				votes.append(pr_val)
		votes_table.append(votes)
	return votes_table

def count_votes(votes_table): # oblicznie liczby glosow
	forest_classification = []
	for votes in votes_table:
		good = votes.count(1)
		bad = votes.count(2)
		if good > bad:
			forest_classification.append(1)
		elif bad > good:
			forest_classification.append(2)
	return forest_classification

def main(argv):
	columns = ['account_status','duration_month','credit_history',
				'purpose','credit_amount','savings',
				'employment_time','installment_rate','sex:status',
				'guarantor','residence_since','property','age',
				'installment_plan','housing','existing_credits',
				'job','maintenance_people','telephone', 'foreign','decision']

	attributes = columns[:-1]
	split_at = 900
	forest_size = [int(argv[0])]
	classification_error = []
	delta = float(argv[1])
	attr_amount = 4

	df = pd.read_csv('data.csv',sep=' ',names=columns,header=None)
	numeric2categoric(df,'duration_month',4)
	numeric2categoric(df,'credit_amount',4)
	numeric2categoric(df,'age',4)

	[df_train,df_test] = split_dataframe(df, split_at)

	for fs in forest_size:

		train_predictions = []
		test_predictions = []

		votes_table = []
		forest_classification = []

		roots = []

		correct_classification = 0
		cost = 0

		weights = [1/df_train.shape[0] for _ in range(df_train.shape[0])]

		print("Building tree's models...")
		for _ in range(fs):
			weights = set_weights(df_train,weights,train_predictions,delta)
			df_bootstrap = bagging(df_train,weights)
			root_node = decision_tree(df_bootstrap,attr_amount,attributes)
			roots.append(root_node)
			train_predictions.append(test_tree(df_train,root_node))

		print("Testing random forest...")
		for root in roots:
			test_predictions.append(test_tree(df_test,root))

		votes_table = ensemble_voting(df_test,test_predictions)
		forest_classification = count_votes(votes_table)

		for i,fc in enumerate(forest_classification):
			if df_test.iloc[[i]]['decision'].values[0] == fc:
				correct_classification += 1
			elif df_test.iloc[[i]]['decision'].values[0] == 1 and fc == 2:
				cost += 1
			elif df_test.iloc[[i]]['decision'].values[0] == 2 and fc == 1:
				cost += 5

		print("Amount of correct classifications:",correct_classification)
		classification_error.append(1 - correct_classification/df_test.shape[0])
		print("Classification error:",classification_error[-1])
		print("Cost:",cost)

if __name__ == '__main__':
	main(sys.argv[1:])
