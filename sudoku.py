# -*- coding: utf-8 -*-

# Sudoku
# Designer: 詹文钰 201320310227

import numpy as np


MAP_TABLE = "0123456789ABCDEFG"

A = 10
B = 11
C = 12
D = 13
E = 14
F = 15
G = 16

"""
Examples:
array_9 = np.array([
    [1,0,0, 3,0,6, 9,0,5], #0
    [0,0,0, 9,0,0, 1,0,0], #1
    [0,0,3, 0,0,0, 0,8,6], #2

    [3,7,0, 0,5,0, 0,0,9], #3
    [0,0,0, 1,3,9, 0,0,0], #4
    [5,0,0, 0,6,0, 0,1,8], #5

    [8,9,0, 0,0,0, 7,0,0], #6
    [0,0,2, 0,0,8, 0,0,0], #7
    [7,0,6, 4,0,5, 0,0,1]  #8
            ])

array_12 = np.array([
    [0,C,0,0, 0,0,0,0, 0,0,0,0], #0
    [3,0,0,1, 0,7,C,0, B,0,0,4], #1
    [8,0,0,B, 0,3,0,0, 0,0,7,5], #2

    [0,0,3,0, 0,1,0,C, A,4,0,0], #3
    [0,0,C,2, A,0,0,0, 7,0,9,0], #4
    [0,0,0,0, 0,5,9,0, 3,C,0,1], #5

    [1,0,8,3, 0,9,7,0, 0,0,0,0], #6
    [0,2,0,6, 0,0,0,A, 4,5,0,0], #7
    [0,0,A,C, 2,0,6,0, 0,1,0,0], #8

    [9,6,0,0, 0,0,4,0, 2,0,0,C], #9
    [A,0,0,5, 0,2,8,0, 9,0,0,6], #10
    [0,0,0,0, 0,0,0,0, 0,0,A,0] #11
            ])

array_16 = np.array([
    [0,0,0,3, 0,7,9,0, 0,C,0,B, 0,0,5,1], #0
    [0,0,B,0, 0,2,0,0, E,1,G,0, 4,6,0,0], #1
    [0,7,5,0, 0,0,B,G, F,A,3,0, 2,0,0,0], #2
    [A,0,0,0, 0,6,4,F, 0,0,5,0, 0,B,9,7], #3

    [0,6,9,0, 0,0,0,0, 5,B,0,0, E,0,7,0], #4
    [8,0,7,0, 0,A,C,0, 0,0,4,0, 0,D,0,2], #5
    [0,D,A,0, 0,0,0,0, 0,F,0,0, 0,0,4,0], #6
    [4,3,0,2, 0,0,0,0, G,0,0,0, 6,0,8,0], #7

    [0,E,0,8, 0,0,0,9, 0,0,0,0, 1,0,D,4], #8
    [0,G,0,0, 0,0,D,0, 0,0,0,0, 0,5,6,0], #9
    [7,0,D,0, 0,5,0,0, 0,E,1,0, 0,8,0,9], #10
    [0,B,0,9, 0,0,1,8, 0,0,0,0, 0,G,3,0], #11

    [C,9,1,0, 0,3,0,0, 6,5,8,0, 0,0,0,A], #12
    [0,0,0,A, 0,4,7,D, 2,3,0,0, 0,9,1,0], #13
    [0,0,6,B, 0,1,G,C, 0,0,A,0, 0,2,0,0], #14
    [3,4,0,0, A,0,E,0, 0,G,9,0, D,0,0,0]  #15
])
"""

def comp(nums, m=9):
    return set(np.arange(1, m+1)) - nums

class SudokuStatus:

    def __init__(self, shape=(9,9)):
        """
        Keyword Arguments:
            shape {tuple} -- the shape of sudoku (default: {(9,9)})
                            in {(9,9), (12, 12), (16, 16)}
        """
        self.record = dict()        #key=>坐标，number=>这个坐标中储存的数
        self.unknown_coord = list() #储存未知坐标(即没有写入值的坐标)

        # other information
        self.shape = shape
        if shape == (9, 9):
            self.room_height = self.room_width = 3 #shape of room
            self.avail_number = set(range(1, 10))
        elif shape == (12, 12):
            self.room_width = 4
            self.room_height = 3
            self.avail_number = set(range(1, 13))
        elif shape == (16, 16):
            self.room_width = self.room_height = 4
            self.avail_number = set(range(1, 17))
        else:
            raise "Incorrect array!"
        
    @staticmethod
    def fromArray(array):
        
        r, c = array.shape
        ss = SudokuStatus((r, c))
        for i in range(r): #定义i的范围在这个数组中
            for j in range(c):
                if array[i,j] != 0: #数组中任何一个数不等于0
                    ss[i,j] = array[i,j]  #记录下已知坐标
                else:
                    ss.unknown_coord.append((i,j))
        return ss

    def copy(self):
        cpy = SudokuStatus(self.shape)
        cpy.record = self.record.copy()
        cpy.unknown_coord = self.unknown_coord.copy()
        return cpy

    def __setitem__(self, coord, val):
        self.record[coord] = val

    def __getitem__(self, coord):
        return self.record.get(coord, 0)

    def __delitem__(self, coord):
        self.record.pop(coord)
        self.unknown_coord.append(coord)


    def debug(self):
        for i in range(side_length):
            for j in range(side_length):
                print("(%d,%d) => %d"%(i,j,self[(i,j)]),end="\t")
            print()


    def __str__(self):
        r, c = self.shape
        return '\n'.join('  '.join(' '.join(str(MAP_TABLE[self[(i,j)]]) for j in range(k*self.room_width, (k+1)*self.room_width)) for k in range(c//self.room_width)) for i in range(r))

    def getCurrentRoom(self, coord):
        row_start = (coord[0] // self.room_height)*self.room_height
        col_start = (coord[1] // self.room_width)*self.room_width
        return range(row_start, row_start+self.room_height), range(col_start, col_start+self.room_width)


    def candidate(self, coord) -> set:
        # C(c)
        # 输入一个坐标，输出这个坐标可以填入的数的候选集

        assert self[coord] == 0, 'The number in coord must be 0!'

        r1, r2 = self.getCurrentRoom(coord)
        curr_room = set(self[row,col] for row in r1 for col in r2)

        r, c = self.shape
        curr_row = set(self[coord[0],col] for col in range(r) if col not in r2)

        #返回给定坐标所在列的已存在的数的list列表
        curr_col = set(self[row,coord[1]] for row in range(c) if row not in r1)
        return self.avail_number - (curr_row | curr_col | curr_room)

    def strict_candidate(self, coord) -> set:
        # S(c) <= C(c)
        # 输入一个坐标，输出这个坐标可以填入的数的严格候选集

        r1, r2 = self.getCurrentRoom(coord)
        room = set()
        for row in r1:
            for col in r2:
                if (row, col) != coord and self[row, col] == 0:
                    room |= self.candidate((row, col))

        r, c = self.shape
        col_ = set()
        for row in range(r):
            if row not in r1 and self[row, coord[1]] == 0:
                col_ |= self.candidate((row, coord[1]))

        row_ = set()
        for col in range(c):
            if col not in r2 and self[coord[0], col] == 0:
                row_ |= self.candidate((coord[0], col))

        cand = self.candidate(coord)
        if len(cand - room) == 1:
            return cand - room
        elif len(cand - row_) == 1:
            return cand - row_
        elif len(cand - col_) == 1:
            return cand - col_
        elif len(cand - (room | row_)) == 1:
            return cand - (room | row_)
        elif len(cand - (room | col_)) == 1:
            return cand - (room | col_)
        elif len(cand - (row_ | col_)) == 1:
            return cand - (row_ | col_)
        elif len(cand - (room| row_ | col_)) == 1:
            return cand - (room| row_ | col_)
        else:
            return cand


def solve(status):
    # unknown: 未知列表
    unknown = status.unknown_coord
    if unknown == []:   # 如果全部坐标都已经填入数字
        print(status)
        return 1
    else:

        for c in sorted(unknown, key=lambda c:len(status.strict_candidate(c))):
            print(c, status.strict_candidate(c))
        raise

        N = 0
        _coord = min(*unknown, key=lambda c:len(status.strict_candidate(c))) # 的可填入数字最少的坐标优先处理
        possible_number = status.strict_candidate(_coord)
        status.unknown_coord.remove(_coord)

        # 遍历当前坐标的所有可能值，然后将其写入数独状态，再递归调用，处理下一个坐标
        for n in possible_number:
            status[_coord] = n
            k = solve(status.copy())
            N += k

        # status.record.pop(curr_coord)
        # status.unknown_coord.append(curr_coord)
        return N

if __name__ == '__main__':
    # initial sudoku state
    array_9 = np.array([
    [0,0,0, 8,9,0, 0,2,0],
    [0,0,9, 0,0,5, 0,0,7],
    [0,5,0, 0,0,0, 3,0,0],

    [0,9,3, 5,0,0, 1,0,0],
    [0,0,0, 1,0,7, 0,0,0],
    [0,0,1, 0,0,6, 8,4,0],

    [0,0,8, 0,0,0, 0,6,0],
    [9,0,0, 6,0,0, 4,0,0],
    [0,0,1, 0,2,8, 0,0,0]])

    sudoku = SudokuStatus.fromArray(array_9)

    N = solve(sudoku)
    print(N)
    print(sudoku)

