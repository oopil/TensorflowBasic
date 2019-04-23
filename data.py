from excel_class import XL, Printer
from sklearn.utils import shuffle
import openpyxl
import numpy as np
import random

class DataLoader():
    def __init__(self, frd):
        self.frd = frd
        self.meta = []
        self.data_tr = []
        self.data_tst = []
        self.label_tr = []
        self.label_tst = []

        self.diag_type = "clinic"
        self.clinic_diag_index = 5
        self.new_diag_index=6

        self.option = ['P', 'T', 'V']
        self.opt_dict_clinic = {
            'AD': 0,
            'CN': 1,
            'aMCI': 2,
            'naMCI': 3
        }
        self.opt_dict_new = {
            'aAD': 0,
            'NC': 1,
            'ADD': 2,
            'mAD': 3
        }

    def data_read(self):
        xl_file_name = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_test.xlsx'
        xl_password = '!adai2018@#'
        xl = openpyxl.load_workbook(xl_file_name, read_only=True)
        ws = xl['Sheet1']
        data = []
        for row in ws.rows:
            line = []
            for cell in row:
                line.append(cell.value)
            data.append(line)
        # print(len(data), len(data[0]))
        # print(data[0])
        # print(data[1])
        return data

    def get_class_name(self, l:list, idx:int) -> list:
        temp = []
        index = 0
        print(len(l))
        for e in l:
            index  += 1
            if index == 1:
                continue
            temp.append(e[idx])

        temp = list(set(temp))
        print('get names of class')
        print(temp)
        return temp

    def count_col_data(self, l:list, type:str, index:int) -> None:
        count = 0
        for e in l:
            if e[index] == type:
                count += 1
        print('it has ', int(count/3), type, 's.')

    def extr_data(self, data, class_num, option_num):
        option = option_num # P T V options
        print('remove some data to use it')
        remove_idx_l = [0,1,4,5,6,7]
        if self.diag_type == "clinic":
            opt_dict = self.opt_dict_clinic
            class_index = 5
        elif self.diag_type == "new":
            opt_dict = self.opt_dict_new
            class_index = 6
        data.remove(data[0])
        # print(data[0])
        label = []
        for i in range(len(data)):
            if i % 3 == option:
                label.append(self.get_class(class_num, data[i][class_index]))
        new_label = []
        new_data = []
        for i in range(len(data)):
            if label[i//3] == -1:
                continue
            new_element = []
            for j in range(len(data[i])):
                if j in remove_idx_l:
                    continue
                new_element.append(data[i][j])
            # if self.is_male(new_element[1]): new_element[1] = 0
            # else : new_element[1] = 1
            # for i in zero_idx_l:
            #     index = length - i - 1
            #     # new_element.erase(0)
            #     del new_element[index]

            if i % 3 == option:
                new_data.append(new_element)
                new_label.append(label[i//3])
        # print(new_data[0])
        # print(label)
        print(len(new_data), len(label))
        return new_data, new_label

    def is_male(self, gender:str):
        if gender == 'M': return True
        elif gender == 'F': return False
        else :
            print('wrong sex is detected.')
            print(gender)
            assert False

    def get_class(self, class_num, class_name) -> int:
        if self.diag_type == "clinic":
            pass
        elif self.diag_type == "new":
            print('this code is not prepared yet.')
            assert False

        if class_num == 2:
            if class_name in ['AD']:
                return 1
            elif class_name in ['CN']:
                return 0
            elif class_name in ['aMCI', 'naMCI']: # do not include MCI patient here about 2 class
                return -1
            else:
                print('wrong class name ', class_name)
                assert False

        # if class_num == 2:
        #     if class_name == 'ADD' or class_name =='mAD':
        #         return 1
        #     elif class_name == 'NC' or class_name == 'mAD':
        #         return 0
        #     else:
        #         print('wrong class name ', class_name)

        elif class_num == 3:
            if class_name in ['AD']:
                return 2
            elif class_name in ['aMCI', 'naMCI']:
                return 1
            elif class_name in ['CN']:
                return 0
            else:
                print('wrong class name ', class_name)
                assert False
        else:
            print('wrong class number ', class_num)
            assert False

    def is_all_zero(self, l:list, idx:int)->bool:
        for e in l:
            if e[idx]:
                return False
        return True

    def diagosis(self):
        pass

def test_something():
    a = [0,1,2,4,8,8,8,8]
    a.insert(3,56)
    print(a)

def dataloader(class_num, option_num):
    loader = DataLoader(2)
    data = loader.data_read()
    return loader.extr_data(data, class_num, option_num)

if __name__ == '__main__':
    # test_something()
    # assert False

    prt = Printer()
    loader = DataLoader(2)
    data = loader.data_read()
    # loader.count_col_data(data, 'AD', 5)
    class_name = loader.get_class_name(data, 6)
    # class_name = loader.get_class_name(data, 5)
    for c in class_name:
        loader.count_col_data(data, c, 6)
    # assert False
    loader.extr_data(data,2)
    # assert False
    # data_tr, label_tr = loader.get_tr_set()
    # data_tst, label_tst = loader.get_tst_set()
    # prt.p_list(data_tr)
    # prt.p_list(label_tr)

def shuffle_2arr(arr1, arr2):
    return shuffle(arr1, arr2, random_state=0)

def valence_class(data, label, class_num):
    length = len(data)
    label_count = [0 for i in range(class_num)]
    label_count_new = [0 for i in range(class_num)]

    for i in sorted(label):
        label_count[i] += 1

    min_count = min(label_count)
    print(min_count)
    new_data = []
    new_label = []
    for i, k in enumerate(label):
        if label_count_new[k] > min_count:
            continue
        new_data.append(data[i])
        new_label.append(label[i])
        label_count_new[k] += 1
    return np.array(new_data), np.array(new_label)

def split_train_test(data, label, option_num, ford_num):
    sample_num = len(data)
    test_num = sample_num // ford_num
    train_num = sample_num - test_num
    # X = np.array(get_random_sample(X, sample_num))
    # Y = np.array(get_random_sample(Y, sample_num))
    X_ = np.array(data)
    Y_ = np.array(label)

    l1 , l2 = len(X_), len(X_[0])
    for i in range(l2):
        col_index = l2 - i - 1
        for j in range(l1):
            if X_[j][col_index]:
                break
            X_ = np.delete(X_, j, 1)


    X_ = normalize(X_)
    X_, Y_ = shuffle_2arr(X_, Y_)
    return X_[:train_num], Y_[:train_num], X_[train_num:sample_num], Y_[train_num:sample_num]

def normalize(X_):
    return (X_-X_.min(0))/X_.max(axis=0)