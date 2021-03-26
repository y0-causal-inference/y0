from y0.dsl import Expression, P, Sum, X, Y, Z



def identify(graph,query):
	P_XY = P(X, Y, Z)
	P_XYZ = P(X, Y, Z)
	return P_XY / Sum[Y](P_XY)