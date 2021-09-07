import copy
import os
import torch

# from sklearn.metrics import classification_report, f1_score
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOBES

from Corpus import words2IOBES , words2IOBES_pub
from Parameters import global_param


def printf(txt, path, display=True):

    file = open(path, "a+")
    print(txt, file=file)
    file.close()
    if display:
        print(txt)


def generate_unique_logpath(logdir, raw_run_name):

    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint:
    def __init__(self, filepath, model, F_type="macro", save=False, corpus="pgx"):

        # self.min_loss = None
        self.best_f = None
        self.filepath = filepath
        self.model = model
        self.save = save
        self.corpus = corpus
        if not save:
            self.best_model = copy.deepcopy(model)
        self.f_type = F_type

    def update(self, pred, Y, epoch, loss, acc, corpus, do_valid=True):
        # print(pred[0])
        # print(Y[0])
        print("|||||||||||| do valid :", do_valid)
        print(corpus)
        if corpus == "pgx":
            f = f1_score(
                y_pred=words2IOBES_pub(pred),
                y_true=words2IOBES_pub(Y),
                average=self.f_type,
                scheme=IOBES,
                mode="strict",
            )
        # torch.save(self.model, self.filepath +"/last_model.pt")
        elif corpus == "pgx_pub":
            f = f1_score(
                y_pred=words2IOBES(pred),
                y_true=words2IOBES(Y),
                average=self.f_type,
                scheme=IOBES,
                mode="strict",
            )
        F_type = global_param.traning_param["F_type"]
        exp_name = global_param.traning_param["exp_tag"]
        machine_name = os.uname()[1]
        file = open(
            exp_name + "result_" + machine_name + "_" + F_type + ".loss_acc", "a+"
        )
        print(str(epoch) + " " + str(loss) + " " + str(acc), file=file)
        file.close()

        printf("---------- epoch : {} ".format(epoch), self.filepath + "/recovery.rec")
        printf(
            " ep : {} Training Loss : {:.4f}, Acc : {:.4f}".format(epoch, loss, acc),
            self.filepath + "/log",
        )
        printf(
            " ep : {} Validation f_mesur : {:.4f}".format(epoch, f),
            self.filepath + "/log",
        )
        printf(
            str(epoch) + " " + str(self.best_f) + "\n",
            self.filepath + "/recovery.rec",
            display=False,
        )

        if (self.best_f is None) or (f > self.best_f) or (not do_valid):

            if not self.save:
                self.best_model = copy.deepcopy(self.model)
            else:
                print("save new model")
                torch.save(self.model, self.filepath + "/best_model.pt")
            # print(corpus)
            # report = classification_report(y_pred= words2IOBES(pred), y_true= words2IOBES(Y),scheme=IOBES,mode='strict')
            if corpus == "pgx":
                report = classification_report(
                    y_pred=words2IOBES(pred),
                    y_true=words2IOBES(Y),
                    scheme=IOBES,
                    mode="strict",
                )
            # torch.save(self.model, self.filepath +"/last_model.pt")
            elif corpus == "pgx_pub":
                report = classification_report(
                    y_pred=words2IOBES_pub(pred),
                    y_true=words2IOBES_pub(Y),
                    scheme=IOBES,
                    mode="strict",
                )
            self.best_f = f

            printf(
                (epoch, "***********************\n************************\n"),
                self.filepath + "/log",
            )
            printf(
                (epoch, "**************** Best f-mesure *************** ", f),
                self.filepath + "/log",
            )
            printf(report, self.filepath + "/log")

    def recovery(self, id):
        self.filepath = "logs/Exp_" + str(id)
        # self.model=torch.load(self.filepath+"/last_model.pt")
        lines = open(self.filepath + "/recovery.rec", "r").read().split("\n")
        index_str = lines[0].split(" ")
        index_str.pop(-1)
        index = [int(i) for i in index_str]

        for i in range(len(lines)):
            if lines[len(lines) - 1 - i].startswith("------"):
                m = lines[len(lines) - 1 - i + 1]
                break

        m = m.split()
        self.best_f = float(m[1])
        epoch = int(m[0])

        return index, epoch - 1

    def get_best(self):
        return self.min_loss, self.best_f

    def get_best_model(self):
        if not self.save:
            return self.best_model
        else:
            return torch.load(self.filepath + "/best_model.pt")


def save_path():
    logdir = "./logs"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logdir = generate_unique_logpath(logdir, "Exp")

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    return logdir
