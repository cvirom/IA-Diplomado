import sys
import time
import numpy as np
from random import shuffle, random, sample, randint
from copy import deepcopy
from math import exp

def get_column_indices(i, type="data index"):
	if type=="data index":
		column=1%9
	elif type=="column index":
		column = i
	indices = [column + 9 * j for j in range(9)]
	return indices
	
def get_row_indices(i, type="data index"):
    if type=="data index":
        row = i // 9
    elif type=="row index":
        row = i
    indices = [j + 9*row for j in range(9)]
    return indices
	
def get_block_indices(k,initialEntries,ignore_originals=False):
	row_offset = (k//3)*3
	col_offset= (k%3)*3
	indices=[col_offset+(j%3)+9*(row_offset+(j//3)) for j in range(9)]
	if ignore_originals:
		#indices = filter(lambda x:x not in initialEntries, indices)
		indices = [x for x in indices if x not in initialEntries]
	return indices

def randomAssign(puzzle, initialEntries):
	for num in range(9):
		block_indices=get_block_indices(num, initialEntries)
		block= puzzle[block_indices]
		zero_indices=[ind for i,ind in enumerate(block_indices) if block[i] == 0]
		to_fill = [i for i in range(1,10) if i not in block]
		shuffle(to_fill)
		for ind, value in zip(zero_indices, to_fill):
			puzzle[ind]=value

def score_board(puzzle):
	score = 0
	for row in range(9): # por cada fila obtiene la cantidad de numeros diferentes
		score-= len(set(puzzle[get_row_indices(row, type="row index")]))
	for col in range(9): # por cada columna obtiene la cantidad de numeros diferentes
		score -= len(set(puzzle[get_column_indices(col,type="column index")]))
	return score

def make_neighborBoard(puzzle, initialEntries):
    new_data = deepcopy(puzzle)
    block = randint(0,8)  # escoje un bloque aleatoriamente
    num_in_block = len(get_block_indices(block,initialEntries,ignore_originals=True)) #cantidad de ´posiciones que se puede mover en el bloque 
    random_squares = sample(range(num_in_block),2)
    square1, square2 = [get_block_indices(block,initialEntries,ignore_originals=True)[ind] for ind in random_squares]
    new_data[square1], new_data[square2] = new_data[square2], new_data[square1]
    return new_data

def showPuzzle(puzzle):
	def checkZero(s):
		if s != 0: return str(s)
		if s == 0: return "0"
	results = np.array([puzzle[get_row_indices(j, type="row index")] for j in range(9)])
	s=""
	for i, row in enumerate(results):
		if i%3==0:
			s +="-"*25+'\n'
		s += "| " + " | ".join([" ".join(checkZero(s) for s in list(row)[3*(k-1):3*k]) for k in range(1,4)]) + " |\n"
	s +="-"*25+''
	print (s)

def sa_solver(puzzle, strParameters):
	""" Simulating annealing solver. 
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...    
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				T0: Initial temperatura
				DR: The decay rate of the schedule function: Ti = T0*(DR)^i (Ti is the temperature at iteration i). For efficiecy it is calculated as Ti = T(i-1)*DR
				maxIter: The maximum number of iterations
	"""
	import shlex
	parameters = {'T0': .5,	'DR': .99999, 'maxIter': 100000} # Dictionary of parameters with default values
	parms_passed = dict(token.split('=') for token in shlex.split(strParameters.replace(',',' '))) # get the parameters from the parameter string into a dictionary
	parameters.update(parms_passed)  # Update  parameters with the passed values
	
	start_time = time.time()
	print ('Simulated Annealing intentará resolver el siguiente puzzle: ')
	showPuzzle(puzzle)
	
	initialEntries = np.arange(81)[puzzle > 0]  # las posiciones no vacias del puzzle
	randomAssign(puzzle, initialEntries)  # En cada box del puzzle asigna numeros aleatorios en pociciones vacias, garantizando que sean los 9 numeros diferentes 
	best_puzzle = deepcopy(puzzle)
	current_score = score_board(puzzle)
	best_score = current_score
	T = float(parameters['T0'])  # El valor inicial de la temperatura
	DR = float(parameters['DR']) # El factor de decaimiento de la temperatura
	maxIter = int(parameters['maxIter']) # El maximo numero de iteraciones
	t = 0
	while (t < maxIter):
		try:
			if (t % 10000 == 0): 
				print('Iteration {},\tTemperaure = {},\tBest score = {},\tCurrent score = {}'.format(t, T, best_score, current_score))
			neighborBoard = make_neighborBoard(puzzle, initialEntries)
			neighborBoardScore = score_board(neighborBoard)
			delta = float(current_score - neighborBoardScore)
			if (exp((delta/T)) - random() > 0):
				puzzle = neighborBoard
				current_score = neighborBoardScore 
			if (current_score < best_score):
				best_puzzle = deepcopy(puzzle)
				best_score = score_board(best_puzzle)
			if neighborBoardScore == -162:   # -162 es el score optimo
				puzzle = neighborBoard
				break
			T = DR*T
			t += 1
		except:
			print("Numerical error occurred. It's a random algorithm so try again.")         
	end_time = time.time() 
	if best_score == -162:
		print ("Solution:")
		showPuzzle(puzzle)
		print ("It took {} seconds to solve this puzzle.".format(end_time - start_time))
	else:
		print("Couldn't solve! ({}/{} points). It's a random algorithm so try again.".format(best_score,-162))
		
def ga_solver(puzzle, strParameters):
	""" Genetic Algorithm solver. 
		puzzle: is a np array of 81 elements. The first 9 are the first row of the puzzle, the next 9 are the second row ...    
		strParameters: a string of comma separated parameter=value pairs. Parameters can be:
				w: Population size
				Cx: Crossover ( single  or uniform )
				m: Mutation rate
				maxGener: The maximum number of generations
	"""
	
	print ("Falta Implementar!")
	
	pass

def default(str):
    return str + ' [Default: %default]'

def readCommand( argv ):
	"""
	Processes the arguments  used to run sudokusolver from the command line.
	"""
	from optparse import OptionParser
	usageStr = """
	USAGE:      python sudokusolver.py <options>
	EXAMPLES:   (1) python sudokusolver.py -p my_puzzle.txt -s sa -a T0=0.5,DR=0.9999,maxIter=100000
	"""
	parser = OptionParser(usageStr)
	parser.add_option('-p', '--puzzle', dest='puzzle', help=default('the puzzle filename'), default=None)
	parser.add_option('-s', '--solver', dest='solver', help=default('name of the solver (sa or ga)'), default='sa')
	parser.add_option('-a', '--solverParams', dest='solverParams', help=default('Comma separated pairs parameter=value to the solver. e.g. (for sa): "T0=0.5,DR=0.9999,nIter=100000"'))
	
	options, otherjunk = parser.parse_args(argv)
	if len(otherjunk) != 0:
		raise Exception('Command line input not understood: ' + str(otherjunk))
	args = dict()
	
	fd = open(options.puzzle,"r+")    # Read the Puzzle file
	puzzle = eval(fd.readline())
	array = []
	for row in puzzle:
		for col in row:
			array.append(col)
	args['puzzle'] = np.array(array)  # puzzle es un vector con todas las filas del puzzle concatenadas (vacios tiene valor 0) 
	args['solver'] = options.solver
	args['solverParams'] =  options.solverParams
	return args	

if __name__=="__main__":
	"""
	The main function called when sudokusolver.py is run from the command line:
	> python sudokusolver.py

	See the usage string for more details.

	> python sudokusolver.py --help
    """
	args = readCommand( sys.argv[1:] ) # Get the arguments from the command line input
	solvers = {'sa': sa_solver,	'ga': ga_solver }  # Dictionary of available solvers
	
	solvers[args['solver']]( args['puzzle'], args['solverParams'] )  # Call the solver method passing the string of parameters
	
	pass

