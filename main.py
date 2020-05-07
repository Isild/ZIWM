import xlsxwriter
import pandas
import numpy as np
import sklearn.feature_selection as feat_select
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

best_conf_matrix = [0, False, 0, np.ndarray, 0]

def main():
    x, y = load_data()
    result = feature_selection(x, y)
    new_list = result[1]

    with xlsxwriter.Workbook('ranking.xlsx') as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(new_list):
            worksheet.write_row(row_num, 0, data)

    layer_widths = [500, 900, 1200]
    momentum = [False, True]
    for width in layer_widths:
        print("Hidden layer width: " + str(width))
        for m in momentum:
            print("Momentum: " + str(m))
            train_evaluate(x, y, width, m)
    print("\n\n\nSUMMARY\n------------------------------------------\n")
    print("Hidden layer width: " + str(best_conf_matrix[0]) + "\nMomentum: " +
          str(best_conf_matrix[1]) + "\nFeatures number: " + str(best_conf_matrix[2]))
    print("Confusion matrix: ")
    print(best_conf_matrix[3])
    print("\nScore: " + str(best_conf_matrix[4]))

    load_data()

def load_data():
    fileNames = ['ang_prct_2.txt', 'ang_prect.txt', 'inne.txt',
                                         'mi_np.txt', 'mi.txt']

    fiX = []
    fiY = []
    classNumber=1

    # pierwsze wczytanie danych z pliku, nie jest to w pętli,
    # ponieważ był problem ze sklejaniem danych o różnych wymiarach,
    # a nie wiem jak zrobić pustą tabele o konkretnych wymiarach
    dataframeX = pandas.read_csv('ang_prct_2.txt', sep="	", header=None)
    dataframeX = dataframeX.T

    number = []
    cC = []
    d1 = []
    x = []
    y = []

    #kolumna z numerami
    for i in range(1, len(dataframeX)+1):
        cC.append(i)
        number.append(classNumber)

    dataframeX.insert(loc=0, column=-1, value=cC)
    dataframeX.insert(loc=0, column=-2, value=number)

    print("Reading files:")
    for f in fileNames:
        print("	", f)
        dataframe = pandas.read_csv(f, sep="	", header=None)
        dataframe=dataframe.T

        number = []
        cC = []
        d1 = []
        x = []
        y = []

        for i in range(1, len(dataframe)+1):
            cC.append(i)
            number.append(classNumber)

        # wstawiam nową kolumnę z numerami,
        # żeby format wporwadzonych danych się zgadzał
        dataframe.insert(loc=0, column=-1, value=cC)
        dataframe.insert(loc=0, column=-2, value=number)
        d1 = dataframe.to_numpy()
        x = d1[:, 2:60]  # features columns; od:do kolumny
        y = d1[:, 0]  # class column; kolumna 0 jako nr klas

        if classNumber != 1:
            # sklejam ze sobą kolejne dane z pliku
            dataframeX=dataframeX.append(dataframe, ignore_index=True)

        classNumber+=1

    print("\n", dataframeX, "\n")

    d1 = dataframeX.to_numpy()
    x = d1[:, 2:61]	# features columns; od:do kolumny
    y = d1[:, 0]	# class column; kolumna 0 jako nr klas

    dataframeX.to_excel("output.xlsx", index=False)
    return x, y.astype(np.int)

def feature_selection(x, y, n_best=59):
    selector = feat_select.SelectKBest(score_func=feat_select.chi2, k=n_best)
    fit = selector.fit(x, y)
    fit_x = selector.transform(x)
    scores = []
    for j in range(n_best):
        scores.append([j, fit.scores_[j]])
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    print("Selected", len(scores), "features")
    return fit_x, scores

def train_evaluate(x, y, hidden_layer_width=900, momentum=True):
    for i in range(1, 60):				# 59 best features
        global best_conf_matrix
        fit_x, _ = feature_selection(x, y, i)
        kf = RepeatedStratifiedKFold(2, 5, random_state=42)	# 5x2cv
        if momentum:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,),
                                max_iter=1000, nesterovs_momentum=True,
                                solver='sgd', verbose=False, random_state=1)
        else:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,),
                                max_iter=1000, solver='sgd', verbose=False,
                                nesterovs_momentum=False, momentum=0,
                                random_state=1)

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

        print("Mean score for feature: " + str(i) +
              " " + str(np.mean(val_acc_features)) + "\n")


if __name__ == "__main__":
    main()
