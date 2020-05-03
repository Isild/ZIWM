import pandas
import numpy as np
import sklearn.feature_selection as feat_select
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
import xlsxwriter

best_conf_matrix = [0, False, 0, np.ndarray, 0]


def main():
    x, y = load_data()
    result = feature_selection(x, y)
    print("dane:\n", x,y)
    print("========================\n")
    print((result[1]))
    new_list = result[1]
    print("długości: ", len(x), len(y))
    print("Score`;\n", len(new_list))
    
    with xlsxwriter.Workbook('ranking.xlsx') as workbook:
    
        worksheet = workbook.add_worksheet()

        for row_num, data in enumerate(new_list):
            worksheet.write_row(row_num, 0, data)
    
    #layer_widths = [500, 900, 1200]
    #momentum = [False, True]
    #for width in layer_widths:
        #print("Hidden layer width: " + str(width))
        #for m in momentum:
            #print("Momentum: " + str(m))
            #train_evaluate(x, y, width, m)
    #print("\n\n\nSUMMARY\n------------------------------------------\n")
    #print("Hidden layer width: " + str(best_conf_matrix[0]) + "\nMomentum: " +
          #str(best_conf_matrix[1]) + "\nFeatures number: " + str(best_conf_matrix[2]))
    #print("Confusion matrix: ")
    #print(best_conf_matrix[3])
    #print("\nScore: " + str(best_conf_matrix[4]))
    
    #load_data()

def load_data():
    #dataframe = pandas.read_excel('ANEMIA.xls')
    #array = dataframe.values
    #print("dane z pliku:\n", array)
    #x = array[:, 2:33]  # features columns; od:do kolumny
    #y = array[:, 0]  # class column; kolumna 0 jako nr klas
    #class_nr = 1.0
    #for it, i in enumerate(y):  # fill empty cells with the class number
        #if np.isnan(i):
            #y[it] = class_nr
        #else:
            #class_nr = i
    #print("anemia:\n", x, y.astype(np.int))
    #return x, y.astype(np.int)
    #print("to co z xls:")
    #print("dane liczbowe:\n",x) 
    #print("kolumna klas\n",y.astype(np.int))
    #print(x, y.astype(np.int))
    
    #dataframe = pandas.read_csv('ang_prect.txt', sep="	", header=None)
    #dataframe = pandas.read_table('ang_prect.txt')
    #print(dataframe)
    #print(len(dataframe[0]))
    #print(len(dataframe))
    #df1 = pandas.DataFrame(data=dataframe)
    #print(dataframe.T)
    #d = dataframe.T
    #print(len(d))
    #cC = []
    #for i in range(1, len(d)+1):
        #cC.append(i)
    #print(cC)

    #d.insert(loc=0, column=-1, value=cC)
    #print(d)
    
    #nasze:
    #array = d
    #print("dane z pliku:\n", array)
    #d1 = d.to_numpy()
    #print(type(d1), "\n\n")
    #x = d1[:, 1:60]  # features columns; od:do kolumny
    #y = d1[:, 0]  # class column; kolumna 0 jako nr klas
    #class_nr = 1.0
    #print(x, y.astype(np.int))
    #return x, y.astype(np.int)
    fileNames = ['ang_prct_2.txt', 'ang_prect.txt', 'inne.txt', 'mi_np.txt', 'mi.txt']
    
    fiX = []
    fiY = []
    classNumber=1
    
    dataframeX = pandas.read_csv('ang_prct_2.txt', sep="	", header=None)
    print("Wczytane dane:\n", dataframeX)
    dataframeX=dataframeX.T
    print("Obrócone dane:\n", dataframeX)
    
    cC = []
    x = []
    y = []
    number = []
    d1 = []
    
    for i in range(1, len(dataframeX)+1):
        cC.append(i)
        number.append(classNumber)
    #print(cC)
    dataframeX.insert(loc=0, column=-1, value=cC)
    print("Tablica z dodanym ponumerowaniem pacjentów:\n", dataframeX)
    dataframeX.insert(loc=0, column=-2, value=number)
    
    
    
    for f in fileNames:
        print("Plik: ", f)
        dataframe = pandas.read_csv(f, sep="	", header=None)
        #print("Wczytane dane:\n", dataframe)
        dataframe=dataframe.T
        #print("Obrócone dane:\n", dataframe)
        
        cC = []
        x = []
        y = []
        number = []
        d1 = []
        
        for i in range(1, len(dataframe)+1):
            cC.append(i)
            number.append(classNumber)
        #print(cC)
        dataframe.insert(loc=0, column=-1, value=cC)
        #print("Tablica z dodanym ponumerowaniem pacjentów:\n", dataframe)
        dataframe.insert(loc=0, column=-2, value=number)
        #print("Tablica z dodanym numerem klasy:\n", dataframe)
        d1 = dataframe.to_numpy()
        x = d1[:, 2:60]  # features columns; od:do kolumny
        y = d1[:, 0]  # class column; kolumna 0 jako nr klas
        #if classNumber == 1:
            #dataframe.to_excel("output.xlsx", index=False) 
            #dataframe.to_excel(writer, startrow=1 , startcol=0)
        #else :
            #dataframe.to_excel("output.xlsx", index=False, header=False,startrow=0 , startcol=0)
            #dataframe.to_excel(writer, startrow=70, startcol=0)
            
        #if len(fiX) == 0:
            #fiX = d1[:, 2:60]
            #fiY = d1[:, 0]
        #else :
            #fiX = np.concatenate([fiX,x])
            #fiX = np.concatenate([fiY,y])
            
        if classNumber != 1:
            dataframeX=dataframeX.append(dataframe, ignore_index=True)
            
        #print("x i y:\n", fiX, fiY)
        classNumber+=1
        #dataframeZ = pandas.DataFrame.copy()
        #dataframeZ = dataframe.copy()
    
    print(dataframeX)
    d1 = dataframeX.to_numpy()
    x = d1[:, 2:61]  # features columns; od:do kolumny
    y = d1[:, 0]  # class column; kolumna 0 jako nr klas
    print(x, y)
    dataframeX.to_excel("output.xlsx", index=False) 
    return x, y.astype(np.int)
    #########################################
    
    #dataframe = pandas.read_csv('ang_prect.txt', sep="	", header=None)
    #print("Wczytane dane:\n", dataframe)
    #dataframe=dataframe.T
    #print("Obrócone dane:\n", dataframe)
    
    #cC = []
    #classNumber=1
    #number = []
    #for i in range(1, len(dataframe)+1):
        #cC.append(i)
        #number.append(classNumber)
    #print(cC)
    #dataframe.insert(loc=0, column=-1, value=cC)
    #print("Tablica z dodanym ponumerowaniem pacjentów:\n", dataframe)
    #dataframe.insert(loc=0, column=-2, value=number)
    #print("Tablica z dodanym numerem klasy:\n", dataframe)
    #d1 = dataframe.to_numpy()
    #x = d1[:, 2:60]  # features columns; od:do kolumny
    #y = d1[:, 0]  # class column; kolumna 0 jako nr klas
    #print(len(np.concatenate([x,x])))
    #print("x i y:\n", x, y)


def feature_selection(x, y, n_best=59):
    print("dl x:", (x), ", y:", (y)) 
    selector = feat_select.SelectKBest(score_func=feat_select.chi2, k=n_best)
    fit = selector.fit(x, y)
    fit_x = selector.transform(x)
    scores = []
    for j in range(59):
        scores.append([j, fit.scores_[j]])
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    print("Score wew:", len(scores))
    return fit_x, scores


def train_evaluate(x, y, hidden_layer_width=900, momentum=True):
    for i in range(1, 8): # 7 best features
        global best_conf_matrix
        fit_x, _ = feature_selection(x, y, i)
        kf = RepeatedStratifiedKFold(2, 5, random_state=42)
        if momentum:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,), max_iter=1000,
                                nesterovs_momentum=True, solver='sgd', verbose=False, random_state=1)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,), max_iter=1000,
                                solver='sgd', verbose=False, momentum=0, random_state=1)
        val_acc_features = []

        for train_index, test_index in kf.split(fit_x, y):
            x_train, x_test = fit_x[train_index], fit_x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            mlp.fit(x_train, y_train)

            prediction = mlp.predict(x_test)
            conf_mat = confusion_matrix(y_test, prediction)
            s = mlp.score(x_test, y_test)
            if best_conf_matrix[4] < s:
                best_conf_matrix = [hidden_layer_width, momentum, i, conf_mat, s]

            val_acc_features.append(s)
        print("Mean score for feature: " + str(i) + " " + str(np.mean(val_acc_features)))


if __name__ == "__main__":
    main()
