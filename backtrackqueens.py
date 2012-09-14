# Michael Ballantyne

import sys

class Queens:
    '''A solution to the n-queens problem in python. The board is represented as a list of row numbers for the queen in each column. Produces all solutions.'''
    def __init__(self, size):
        self.size = size
    
    def isUnderAttack(self, positions):
        row = positions[-1]
        column = len(positions)-1
        for i in range(column):
            if positions[i] == row:
                return True
            if positions[i] == row - (column -i):
                return True
            if positions[i] == row + (column-i):
                return True
        
        return False
    
    def board_as_string(self, positions):
        result = ""
        for row in range(len(positions)):
            for column in range(len(positions)):
                if positions[column] == row:
                    result = result + "Q"
                else:
                    result = result + "_"
                result += " "
            result += "\n"
        return result
            
    def placeQueen(self, positions = []):
        if len(positions) != 0 and self.isUnderAttack(positions):
            return []
        elif len(positions) >= self.size:
            return positions
        else:
            results = [];
        
            for row in range(self.size):
                positions.append(row)
                result = self.placeQueen(list(positions))
                if len(result) > 0:
                    return result
                else:
                    positions.pop()
            
            return results
            
def main():
    try:
        board_size = int(sys.argv[1])
    except:
        print "Usage: python singlequeen.py boardsize"
        exit(1)

    queens = Queens(board_size)
    board = queens.placeQueen()
    print queens.board_as_string(board)

if __name__ == "__main__":
    main()
