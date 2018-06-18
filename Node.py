class Node(object):
	def __init__(self,attr,values):
		self.attr = attr
		self.values = values
		self.children = []
		self.leaves = []
	
	def set_child(self,ctg,child_node): 
		self.children.append([ctg,child_node])

	def set_leave(self,value,c):
		self.leaves.append([value,c])

	def __iter__(self):
		return self

	def next_node(self,val):
		for child in self.children:
			if child[0] == val: # w child[0] zawiera sie wartosc atrybutu
				return child[1] # w child[1] zawiera sie atrybut potomny
				
	def get_class(self,val):
		for leave in self.leaves:
			if leave[0] == val: # w leave[0] zawiera sie wartosc atrybutu
				return leave[1] # w leave[1] zawiera sie lisc, czyli klasa(1-good,2-bad)