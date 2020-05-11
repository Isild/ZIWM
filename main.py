import xlsxwriter
import pandas
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as feat_select
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier

best_conf_matrix = [0, False, 0, 0, np.ndarray, 0]

def main():
    x, y = load_data()
    result = feature_selection(x, y)
    new_list = result[1]
    if len(sys.argv) > 1:
        max_features = int(sys.argv[1])
    else:
        max_features = 59
    if (max_features > 59 or max_features < 1):
        raise ValueError("Must check for at least one feature and max 59")

    with xlsxwriter.Workbook('ranking.xlsx') as workbook:
        worksheet = workbook.add_worksheet()
        for row_num, data in enumerate(new_list):
            worksheet.write_row(row_num, 0, data)

    layer_widths = [200, 800, 1400]
    momentum_coef = [0, 0.35, 0.65, 0.95]
    feature_cnt = list(range(1, max_features + 1))
    for width in layer_widths:
        print("Hidden layer width: " + str(width))
        scores = []
        workbook = xlsxwriter.Workbook("scores" + str(width) + ".xlsx")
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, "Feature Count")
        worksheet.write_column(1, 0, feature_cnt)
        col = 1
        for mc in momentum_coef:
            print("Momentum coefficient: " + str(mc))
            if (mc == 0):
                print("Momentum: " + str(False))
                scores = train_evaluate(x, y, width, False, mc, max_features)
                data_label = "M: NO"
            else:
                print("Momentum: " + str(True))
                scores = train_evaluate(x, y, width, True, mc, max_features)
                data_label = "M: " + str(mc)
            worksheet.write(0, col, "M: " + str(mc))
            worksheet.write_column(1, col, scores)
            plt.plot(feature_cnt, scores, label=data_label,
                     linewidth=1, marker='o', markersize=5)
            col += 1

        plt.title("Hidden layer width: " + str(width))
        plt.xlabel('Feature Count')
        plt.ylabel('Mean Score')
        plt.xlim([0, max_features + 1])
        plt.ylim([0, 1])
        plt.grid(True)
        plt.grid(which='both')
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)
        plt.legend()
        plt.savefig("W_" + str(width) + ".png", dpi=600)
        plt.clf()
        workbook.close()

    summaries = "Hidden layer width: " + str(best_conf_matrix[0]) + \
                "\nMomentum: " + str(best_conf_matrix[1]) + \
                "\nMomentum coef: " + str(best_conf_matrix[2]) + \
                "\nFeatures number: " + str(best_conf_matrix[3]) + \
                "\nConfusion matrix:\n" + str(best_conf_matrix[4]) + \
                "\nScore: " + str(best_conf_matrix[5])
    print("SUMMARY\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
    print(summaries)
    summary = open("summary.txt","w+")
    summary.write(summaries)
    summary.close()


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

def train_evaluate(x, y, hidden_layer_width=900, momentum=True, momentum_coef=0.9, max_features=59):
    # test first 1, 2 ... 'max_features' best features
    scores = []
    for i in range(1, max_features + 1):
        global best_conf_matrix
        fit_x, _ = feature_selection(x, y, i)
        kf = RepeatedStratifiedKFold(2, 5, random_state=42)	# 5x2cv
        if momentum:
            mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer_width,),
                                max_iter=1000, nesterovs_momentum=True,
                                solver='sgd', verbose=False, random_state=1,
                                momentum=momentum_coef)
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
            if best_conf_matrix[5] < s:
                best_conf_matrix = [hidden_layer_width, momentum,
				    momentum_coef, i, conf_mat, s]
            val_acc_features.append(s)				# 5x2cv scores

        print("Mean score for " + str(i) + " features: " +
              str(np.mean(val_acc_features)) + "\n")
        scores.append(np.mean(val_acc_features))
    return scores

if __name__ == "__main__":
    main()
