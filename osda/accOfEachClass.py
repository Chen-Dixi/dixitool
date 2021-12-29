# 用来计算开放集领域自适应的准确率，实验结果按照论文Open Set Domain Adaptation by Backpropagation中的实验设定计算
import numpy as np
import torch

# 按照 [总的准确率判断 ，各个类的准确率求平均值，某个类别的准确率] 衡量最好结果
ALLOWED_BEST_METRICS = ['total_acc', 'mean_acc_overclasses','category']
"""
The average accuracy of all classes including the unknown one is denoted as OS.
Accuracy measures only on the known classes of the target domain is denoted as OS*.
"""
ALLOWED_RESULT_TYPE = ['total','OS','OS*','category']

class AccCalculatorForEveryClass(object):
    
    best_result = 0.0
    best_method = 'total_acc'# 判断best_model的方式
    
    
    def __init__(self,num_classes,eps=1e-7,last_epoch=-1):
        self.classes = []    
        self.eps = eps
        
        self.num_classes = num_classes
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)
        self.best_corrects = np.zeros(self.num_classes)
        self.best_totals = np.zeros(self.num_classes)
        self.best_category_id = 0 #int
        self.last_epoch=last_epoch
        self.best_epoch=0
        self.header = ""        
        
        for i in range(num_classes):
            self.classes.append(str(i))

        self.step(last_epoch)

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
    def update(self, pred, target,binary_sigmoided=False):
        
        # if pred is binary guess
        if pred.size(1) == 1:
            if not binary_sigmoided:
                pred = torch.sigmoid(pred)
            pred = pred.ge(0.5).float().view(-1) # still tensor; size=(pred.size(0),1), dtype=torch.uint8
            
        else:
            pred = torch.argmax(pred,dim=1) #size = (pred,size(0),) 1-dimensionx    
        
        for cur_cls in range(self.num_classes):
            target_mask = (target == cur_cls).byte() 
            pred_mask = (pred == cur_cls).byte()
            self.totals[cur_cls] += target_mask.float().sum().item() #sum()表示这个类别的总数
            self.corrects[cur_cls] +=  (pred_mask & target_mask).float().sum().item() #sum()表示这个batch里面类别cur_cls有多少预测正确
    """
    'The average accuracy of all classes including the unknown one is denoted as OS.
     Accuracy measures only on the known classes of the target domain is denoted as OS*.'
    """
    def get(self, result_type='total', category_id=None):
        if not result_type in ALLOWED_RESULT_TYPE:
            raise NotImplementedError("type not allowed")
        if result_type == 'total':
            return 100.* self.corrects.sum()/(self.totals.sum()+self.eps)
        elif result_type == 'OS':
            return 100.* (self.corrects/(self.totals+self.eps)).mean()
        elif result_type == 'OS*':
            return 100.* (self.corrects[:-1]/(self.totals[:-1]+self.eps)).mean()
        elif result_type == 'category':
            if category_id is None:
                raise ValueError("category method must specify a category ID")
            return 100.*self.corrects[category_id]/(self.totals[category_id]+self.eps)

    def reset(self):
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)

    def reset_all(self):
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)
        self.best_result = 0.0        

    def reset_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.corrects = np.zeros(self.num_classes)
        self.totals = np.zeros(self.num_classes)

    def set_best_method(self, best_method='total_acc',category_id=None):
        """
        category_id 从0开始数
        """
        if not best_method in ALLOWED_BEST_METRICS:
            raise NotImplementedError("method not allowed")

        self.best_method = best_method
        self.best_result = 0.0

        if best_method == 'category' and category_id is None:
            raise ValueError("category method must specify a category ID")

        if best_method == 'category':
            self.best_category_id = category_id



    def print_result(self, save_best=False):
        self.print_info()
        total_acc = 100.* self.corrects.sum()/(self.totals.sum()+self.eps)
        mean_acc_overclasses = 100.* (self.corrects/(self.totals+self.eps)).mean()
        known_acc_average = 100.* (self.corrects[:-1]/(self.totals[:-1]+self.eps)).mean()
        print("ALL Accuracy:{:.4f}%".format(total_acc))
        print("OS Accuracy over all classes:{:.4f}%".format(mean_acc_overclasses))
        print("OS* Accuracy over known classes:{:.4f}%".format(known_acc_average))
        for i in range(self.num_classes):
            class_name = self.classes[i]
            print("{}:{:.4f}%".format(class_name, 100.*self.corrects[i]/(self.totals[i]+self.eps)))

        if save_best:
            is_best = False
            if self.best_method=='total_acc':
                is_best = total_acc > self.best_result
                self.best_result = max(total_acc, self.best_result)
                
            elif self.best_method=='mean_acc_overclasses':
                is_best = mean_acc_overclasses > self.best_result
                self.best_result = max(mean_acc_overclasses, self.best_result)

            elif self.best_method == 'category':

                the_category_acc = 100. * self.corrects[self.best_category_id]/(self.totals[self.best_category_id]+self.eps)
                is_best = the_category_acc > self.best_result
                self.best_result = max(the_category_acc, self.best_result)

            if is_best:
                self.best_corrects = self.corrects.copy()
                self.best_totals = self.totals.copy()
                self.best_epoch = self.last_epoch

    def print_best_result(self):
        self.print_header()
        self.print_best_epoch()
        total_acc = 100.* self.best_corrects.sum()/(self.best_totals.sum()+self.eps)
        mean_acc_overclasses = 100.* (self.best_corrects/(self.best_totals+self.eps)).mean()
        known_acc_average = 100.* (self.best_corrects[:-1]/(self.best_totals[:-1]+self.eps)).mean()
        print("Total Accuracy:{:.4f}%".format(total_acc))
        print("Mean Accuracy over all classes:{:.4f}%".format(mean_acc_overclasses))
        print("OS* Accuracy over known classes:{:.4f}%".format(known_acc_average))
        for i in range(self.num_classes):
            class_name = self.classes[i]
            print("{}:{:.4f}%".format(class_name, 100.*self.best_corrects[i]/(self.best_totals[i]+self.eps)))
    
    def set_header_info(self, info=None):
        if info == None:
            info="==========     Result      ============= "
        self.header = info

    def print_info(self):
        self.print_header()
        self.print_epoch()

    def print_epoch(self):
        print("Current Epoch: {}".format(self.last_epoch))
    def print_best_epoch(self):
        print("Best Epoch: {}".format(self.best_epoch))
    def print_header(self):
        print(self.header)

    def step(self,epoch=None):
        if epoch==None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
