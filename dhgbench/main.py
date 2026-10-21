import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'dhgbench'))
from parameter_parser import parameter_parser,method_config,set_task_args


def should_defer_model_preprocessing(args):
    return (
        args.task_type == 'edge_pred'
        and args.edge_pred_protocol == 'observed'
    )


def run_single(args):
    args = method_config(args) 
    args = set_task_args(args) 
    if getattr(args, 'ene_batch_output', None):
        from lib_utils.ene_batch import configure_worker
        configure_worker(args)
    # Batch scheduling stays independent of the model/GPU dependencies.
    from lib_utils.exp_agent import ExpAgent
    from lib_models.HNN.preprocessing import algo_preprocessing
    from lib_dataset.data_base import HyperDataset
    from lib_dataset.preprocessing import data_processing
    from lib_dataset.data_perturbation import perturbation


    data=HyperDataset(args) 

    if args.task_type == 'hg_cls':
        data = data.multi_hypergraphs 
    else:
        data = data_processing(args,data)
        data._initialization_()

        if args.is_perturbed:
            if isinstance(args.pert_p,str):
                args.pert_p = eval(args.pert_p)
            if args.pert_mode not in ['spar_label','flip_label']:
                print('Robustness Perturbation for Structure and Feature')
                if args.is_poison:
                    print('Performing Poison Attack!')
                    data = perturbation(data,mode=args.pert_mode,p=args.pert_p,masks=None)
                else:
                    print('Performing Evasion Attack!')
                    evasion_data = perturbation(data,mode=args.pert_mode,p=args.pert_p,masks=None)
            else:
                print('Robustness Perturbation for Supervision Signal')
                if args.is_poison == False:
                    raise ValueError('Label attack is expected to be the poison attack!')

        if not should_defer_model_preprocessing(args):
            data = algo_preprocessing(data,args)

            if args.is_perturbed and not args.is_poison:
                evasion_data = algo_preprocessing(evasion_data,args)
                data.evasion_data = evasion_data # store evasion data
    
    agent = ExpAgent(args)
    agent.running(args.task_type,data)


if __name__ == '__main__':
    args = parameter_parser()
    if args.all_datasets or args.datasets is not None:
        from lib_utils.ene_batch import run_batch
        run_batch(args)
    elif args.ene_batch_output:
        from lib_utils.ene_batch import run_worker
        sys.exit(run_worker(args, run_single))
    else:
        run_single(args)
