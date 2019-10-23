# 用来计算开放集领域自适应的准确率，实验结果按照论文Open Set Domain Adaptation by Backpropagation中的实验设定计算
import numpy as np
import torch

class AccCalculatorForEveryClass(object):
    
    def __init__(self,num_classes,eps=1e-7):
        self.classes = []    
        self.eps = eps
        
        self.num_classes = num_classes
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)

        for i in range(num_classes):
            self.classes.append(str(i))

    def set_classes_name(self, classes):
        classes_list = list(classes)
        if len(classes_list)==0:
            raise ValueError("calculator got an empty class list")
        if not isinstance(classes_list[0], str):
            raise ValueError("the element of the list should be str")

        self.classes = classes_list

        if len(classes) != self.num_classes:
            print("Warning, num_classes doesn't equal to length of list")

    # pred target 都是torch的tensor
    def update(self, pred, target):
        
        # if pred is binary guess
        if pred.size(1) == 1:
            pred = torch.sigmoid(pred)
            pred = pred > 0.5 # still tensor; size=(pred.size(0),1), dtype=torch.uint8
        else:
            pred = torch.argmax(pred,dim=1)

        for cur_cls in range(self.num_classes):
            target_mask = (target == cur_cls).byte() 
            pred_mask = (pred == cur_cls).byte()
            self.totals[cur_cls] += target_mask.float().sum().item() #sum()表示这个类别的总数
            self.corrects[cur_cls] +=  (pred_mask & target_mask).float().sum().item() #sum()表示这个batch里面类别cur_cls有多少预测正确

    def get(self, ignore_last_class=False):
        if not ignore_last_class:
            return 100.* self.corrects.sum()/(self.totals.sum()+self.eps)
        else:
            return 100.* self.corrects[:-1].sum()/(self.totals[:-1].sum()+self.eps)


    def reset(self):
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)

    def reset_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)

    def print_result(self, save_best=False):
        print("Total Accuracy:{:.4f}%".format(100.* self.corrects.sum()/(self.totals.sum()+self.eps)))
        print("Mean Accuracy over all classes:{:.4f}%".format(100.* (self.corrects/(self.totals+self.eps)).mean()))
        for i in range(self.num_classes):
            class_name = self.classes[i]
            print("{}:{:.4f}%".format(class_name, 100.*self.corrects[i]/(self.totals[i]+self.eps)))

    """
    save best parameters
    """