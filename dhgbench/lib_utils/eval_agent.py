from collections import defaultdict
from lib_utils.metrics import *
from lib_models import _semi_methods_
from lib_dataset import _fair_datasets_

class Evaluator:
    
    def __init__(self,args, **kwargs):
        """
        Training pipline for different kinds of models
        """
        self.args = args
        self.device=args.device
        
    @torch.no_grad()
    def node_cls_evaluation(self, model, data, masks, result=None):

        if result is not None:
            logits = result
        else:
            model.eval()
            logits,_ = model(data) 

        accs = accuracy(logits, data.y, masks)
        metrics_dict = {'acc': accs}

        return metrics_dict
    
    @torch.no_grad()
    def edge_pred_evaluation(self, model, data, batch_loaders, result=None):
        train_metrics, val_metrics, test_metrics = evaluate_edge(model,data,batch_loaders)
        return train_metrics, val_metrics, test_metrics
    
    @torch.no_grad()
    def hg_cls_evaluation(self, model, batch_loaders, result=None):
        result = evaluate_hypegraph(model,batch_loaders,self.args)
        return result
    
    @torch.no_grad()
    def node_cls_with_fairness(self, model, data, masks, result=None):
        
        if result is not None:
            logits = result
        else:
            model.eval()
            logits,_ = model(data) # logits
        
        accs = accuracy(logits, data.y, masks)
        
        f1s = f1_scores(logits,data.y,masks)
        
        aucs = auc_rocs(logits,data.y,masks)
        
        dps,eos = fairness(logits,data.y,masks,data.sens)

        metrics_dict={'acc':accs,'f1_score':f1s,'auc':aucs,'parity':dps,'equality':eos}
        return metrics_dict

    def evaluate(self,model,data,seed_split=None,task_type=None,verbose=False):
        
        metrics_dict=defaultdict(list) 
        
        if self.args.method in _semi_methods_:
            
            if task_type == 'node_cls':
                
                if data.name in _fair_datasets_:
                    metrics=self.node_cls_with_fairness(model, data, seed_split)
                else:
                    metrics=self.node_cls_evaluation(model, data, seed_split)
                    
                if verbose:
                    for m in metrics:
                        print(f'train_{m}: {metrics[m][0]:.2f}, valid_{m}: {metrics[m][1]:.2f}, test_{m}: {metrics[m][2]:.2f} ')
                
                metrics_dict = metrics_dict |  metrics     

            elif task_type == 'edge_pred':
                
                metrics_dict = defaultdict(None)
                train_metrics, val_metrics, test_metrics = self.edge_pred_evaluation(model,data,seed_split)
                
                if verbose:
                    edge_evaluation_printer(train_metrics, val_metrics, test_metrics)
                
                metrics_dict['train'],metrics_dict['val'],metrics_dict['test'] = train_metrics, val_metrics, test_metrics
            
            elif task_type == 'hg_cls':
                
                result = self.hg_cls_evaluation(model,seed_split)
                
                if verbose:
                    hg_evaluation_printer(result)
                
                metrics_dict = result 
            
        else:
            raise NotImplementedError
        
        return metrics_dict
